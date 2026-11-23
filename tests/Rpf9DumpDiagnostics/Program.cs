using System.Buffers.Binary;
using RpfPreview.Core;

Expect([], 4, "unrecognized-header");
Expect([], RageModelDecoder.MaxGraphicsBytes + 1, "graphics-size-limit");

// A synthetically truncated resource prefix containing a system-page pointer.
var prefix = new byte[16];
BinaryPrimitives.WriteUInt64LittleEndian(prefix.AsSpan(8), 0x50000080);
Expect(prefix, 16, "truncated-system");

// Recognizable drawable header with a deliberately malformed page table.
var header = new byte[0x100];
foreach (int at in new[] { 8, 16, 80, 168 })
    BinaryPrimitives.WriteUInt64LittleEndian(header.AsSpan(at), 0x500000c0);
BinaryPrimitives.WriteSingleLittleEndian(header.AsSpan(44), 2);
foreach (int at in new[] { 64, 68, 72 }) BinaryPrimitives.WriteSingleLittleEndian(header.AsSpan(at), 1);
Expect(header, 16, "invalid-or-unsupported-layout");
Console.WriteLine("Model decode diagnostic smoke tests passed.");

// Util.WriteAtomic: outputs appear under their final name only once complete.
string dir = Directory.CreateTempSubdirectory("rpf9dump-test").FullName;
try
{
    string path = Path.Combine(dir, "a.dds");
    Util.WriteAllBytes(path, [1, 2, 3]);
    Check(File.ReadAllBytes(path).SequenceEqual(new byte[] { 1, 2, 3 }), "written file has the data");
    Check(Directory.GetFiles(dir).Length == 1, "no .partial left after a write");

    Util.WriteAllText(path, "new");
    Check(File.ReadAllText(path) == "new", "an existing file is replaced");

    string failed = Path.Combine(dir, "b.dds");
    try { Util.WriteAtomic(failed, t => { File.WriteAllBytes(t, [9]); throw new IOException("disk full"); }); Check(false, "error propagates"); }
    catch (IOException) { }
    Check(!File.Exists(failed) && Directory.GetFiles(dir, "b.dds*partial").Length == 0,
        "a failed write leaves neither file nor a .partial");

    // A stopped run's leftover .partial must not block a writer that opens with FileMode.CreateNew.
    string stale = Path.Combine(dir, "c.xma.wav");
    File.WriteAllBytes(stale + ".partial", [7]);
    Util.WriteAtomic(stale, t => { using var f = new FileStream(t, FileMode.CreateNew); f.WriteByte(5); });
    Check(File.ReadAllBytes(stale).SequenceEqual(new byte[] { 5 })
         && !File.Exists(stale + ".partial")
         && Directory.GetFiles(dir, "c.xma.wav*partial").Length == 0,
         "stale .partial is replaced");
}
finally { Directory.Delete(dir, true); }
Console.WriteLine("Atomic write tests passed.");

// Scan counts round-trip through the index cache's coverage text, including older culture-formatted caches.
Check(ScanCounts.Candidates(ScanCounts.Coverage(183801, 149811)) == 183801, "invariant coverage");
Check(ScanCounts.Candidates("rpf9dump full scan, 183,801 candidates, 149,811 recovered") == 183801, "comma-grouped coverage");
Check(ScanCounts.Candidates("rpf9dump full scan, 183\u00A0801 candidates, 149\u00A0811 recovered") == 183801, "space-grouped coverage");
// Every culture's N0 output must read back (de-CH uses U+2019, ar-* U+066C, nqo U+060C, ff-Adlm U+2E41).
foreach (var culture in System.Globalization.CultureInfo.GetCultures(System.Globalization.CultureTypes.AllCultures))
{
    string old = string.Format(culture, "rpf9dump full scan, {0:N0} candidates, {1:N0} recovered", 183801, 149811);
    Check(ScanCounts.Candidates(old) == 183801, "coverage written under " + culture.Name);
}
Check(ScanCounts.Candidates("App scan of the first 16 MiB") is null, "coverage written by another tool");
Check(ScanCounts.Candidates(null) is null, "no cache");
Console.WriteLine("Scan count tests passed.");

// Util.MeshKey reads the model/geometry pair the decoder puts in a mesh name; the geometry index of a dehydrated
// BLAS leaf counts geometries inside one model, so the pair decides which list a signature is matched against.
Check(Util.MeshKey("model_0_geometry_3") == (0, 3), "mesh key model 0 geometry 3");
Check(Util.MeshKey("model_12_geometry_0") == (12, 0), "mesh key model 12 geometry 0");
Check(Util.MeshKey("model_7_geometry_11") == (7, 11), "mesh key two-digit geometry");
foreach (string other in new[] { "", "model", "model_0_geometry", "model_0_geometry_1_2", "Model_0_geometry_1",
             "model_x_geometry_1", "model_0_geometry_1a", "model__geometry_1", "model_0_Geometry_1", "model_-1_geometry_1" })
    Check(Util.MeshKey(other) == (-1, -1), "not a mesh key: " + other);
Console.WriteLine("Mesh key tests passed.");

// RecoveredResource.Read turns the recovery step's block list back into the resource's bytes: a window slice has no
// archive document, so this is the only reader models and textures have there.
{
    byte[] system = System.Text.Encoding.ASCII.GetBytes("system segment, zlib");
    var gfx = new byte[64];
    for (int i = 0; i < gfx.Length; i++) gfx[i] = (byte)(0x40 + i);
    using var compressed = new MemoryStream();
    using (var deflate = new System.IO.Compression.ZLibStream(compressed, System.IO.Compression.CompressionLevel.SmallestSize, true))
        deflate.Write(system);
    byte[] zblock = compressed.ToArray();

    string segmentDir = Directory.CreateTempSubdirectory("rpf9dump-segments").FullName;
    try
    {
        string path = Path.Combine(segmentDir, "window.bin");
        var file = new byte[32 + zblock.Length + 16 + gfx.Length];
        Array.Fill(file, (byte)0xAA);
        zblock.CopyTo(file, 32);
        gfx.CopyTo(file, 32 + zblock.Length + 16);
        File.WriteAllBytes(path, file);
        ArchiveEntry Entry(params ArchiveSegment[] segments) => new("recovered/models/x_at_0.r9modeldata",
            segments.Sum(s => s.StoredSize), segments.Sum(s => s.Size), 32, false, false, false, "Recovered model", 0) { Segments = segments };
        var entry = Entry(new ArchiveSegment(32, zblock.Length, system.Length),
                          new ArchiveSegment(32 + zblock.Length + 16, gfx.Length, gfx.Length) { IsRaw = true });
        byte[] read = RecoveredResource.Read(path, entry);
        Check(read.SequenceEqual(system.Concat(gfx)), "one compressed and one raw block, in stored order");

        // A block that reaches past the file is an error, not a short read.
        var pastEnd = Entry(new ArchiveSegment(32, file.Length, system.Length));
        try { RecoveredResource.Read(path, pastEnd); Check(false, "block past the end is rejected"); }
        catch (InvalidDataException) { }
        // A raw block reads its own bytes; the 6-byte minimum is the smallest block the recovery step records.
        var shortRaw = Entry(new ArchiveSegment(32 + zblock.Length + 16, 8, 8) { IsRaw = true });
        Check(RecoveredResource.Read(path, shortRaw).SequenceEqual(gfx[..8]), "a raw block reads its own bytes");
        var tooSmall = Entry(new ArchiveSegment(32 + zblock.Length + 16, 4, 4) { IsRaw = true });
        try { RecoveredResource.Read(path, tooSmall); Check(false, "a block under six bytes is rejected"); }
        catch (InvalidDataException) { }
        // An entry that records no blocks is rejected rather than returning an empty resource.
        var oversize = new ArchiveEntry("recovered/models/x.r9modeldata", 1, 1, 32, false, false, false, "Recovered model", 0);
        try { RecoveredResource.Read(path, oversize); Check(false, "an entry without blocks is rejected"); }
        catch (InvalidDataException) { }
    }
    finally { Directory.Delete(segmentDir, true); }
}
Console.WriteLine("Recovered segment tests passed.");

// Only a full scan replaces rpf9dump's own: rpf9-explorer saves partial model indexes while indexing and on cancel.
Check(!ScanCounts.IsFullScan(Index("Dictionary-aware model scan in progress \u00B7 1,001/183,801 containers validated.", false)), "GUI index saved while indexing");
Check(!ScanCounts.IsFullScan(Index("Dictionary-aware model scan canceled. Validated models are retained; discovery will resume on reopen.", false)), "GUI index saved on cancel");
Check(ScanCounts.IsFullScan(Index("Full 1,234,567-byte model-container scan \u00B7 149,811 supported models including drawable dictionary children.", true)), "GUI index marked complete");
Check(ScanCounts.IsFullScan(Index(ScanCounts.Coverage(183801, 149811), true)), "rpf9dump index");
Check(ScanCounts.IsFullScan(Index(ScanCounts.Coverage(183801, 149811), false)), "rpf9dump index saved without the complete mark");
Check(ScanCounts.IsFullScan(Index(string.Format(System.Globalization.CultureInfo.GetCultureInfo("de-CH"), "rpf9dump full scan, {0:N0} candidates, {1:N0} recovered", 183801, 149811), false)), "culture-formatted rpf9dump index");
Check(!ScanCounts.IsFullScan(null), "no cached index");
Console.WriteLine("Cached index tests passed.");

// External BCPack history tables (BcpackHistory, rpf9dump --history): the same JSON tools/bcpack_history_calib.py writes
// and tools/bcpack.py reads. The chunks are one-block BCPack streams from tests/test_bcpack.py's synthetic encoder (zero
// deltas, so a block's endpoints are its history entry), and every expected result is what tools/bcpack.py returns for
// the same chunk and table: decoded bytes, then "|" and the provisional labels, or "refused:" and the message.
string ListForm = System.Text.Json.JsonSerializer.Serialize(new Dictionary<string, object> { ["bc1.json"] = new Dictionary<string, object>
    { ["codec"] = "bc1", ["replace"] = true, ["confirmed"] = Enumerable.Range(0, 256).Select(i => new[] { i % 32, i % 64, 0, 0, 0, i % 32 }).ToArray() } });

var decodeCases = new (string Name, Func<byte[]> Chunk, int Codec, int Size, string? Docs, bool Folder, bool Provisional, string Expected)[]
{
    ("bc1 provisional entry, no tables, strict", Bc1Entry6, 1, 8, null, true, false, "refused:BC1 block 0 references unconfirmed built-in history entry 6"),
    ("bc1 provisional entry, no tables, provisional", Bc1Entry6, 1, 8, null, true, true, "0B63C639AAAAAAAA|BC1 history 6"),
    ("bc1 entry confirmed by table", Bc1Entry6, 1, 8, "{\"bc1.json\": {\"codec\": \"bc1\", \"confirmed\": {\"6\": [1, 2, 3, 4, 5, 6]}, \"provisional\": {}}}", true, false, "4308A620AAAAAAAA"),
    ("bc1 table provisional overrides built-in provisional", Bc1Entry6, 1, 8, "{\"bc1.json\": {\"codec\": \"bc1\", \"provisional\": {\"6\": [31, 63, 31, 0, 1, 2]}}}", true, true, "FFFF2200AAAAAAAA|BC1 history 6"),
    ("bc1 table provisional never overrides built-in confirmed", Bc1Entry255, 1, 8, "{\"bc1.json\": {\"codec\": \"bc1\", \"provisional\": {\"255\": [31, 63, 31, 0, 1, 2]}}}", true, false, "01000000AAAAAAAA"),
    ("bc1 replace drops built-in confirmed", Bc1Entry255, 1, 8, "{\"bc1.json\": {\"codec\": \"bc1\", \"replace\": true}}", true, true, "refused:BC1 block 0 references unknown built-in history entry 255"),
    ("bc1 replace keeps table provisional", Bc1Entry255, 1, 8, "{\"bc1.json\": {\"codec\": \"bc1\", \"replace\": 1, \"provisional\": {\"255\": [3, 4, 5, 6, 7, 8]}}}", true, false, "refused:BC1 block 0 references unconfirmed built-in history entry 255"),
    ("bc1 replace keeps table provisional, provisional", Bc1Entry255, 1, 8, "{\"bc1.json\": {\"codec\": \"bc1\", \"replace\": 1, \"provisional\": {\"255\": [3, 4, 5, 6, 7, 8]}}}", true, true, "8518E830AAAAAAAA|BC1 history 255"),
    ("bc1 256-list form", Bc1Entry6, 1, 8, ListForm, true, false, "C0300600AAAAAAAA"),
    ("bc1 combined file section", Bc1Entry6, 1, 8, "{\"history.json\": {\"bc1\": {\"confirmed\": {\"6\": [9, 9, 9, 9, 9, 9]}}, \"bc7\": {\"historyEndpoints\": {}}}}", false, false, "29492949AAAAAAAA"),
    ("bc1 combined file without bc1 keeps built-in", Bc1Entry6, 1, 8, "{\"history.json\": {\"bc4\": {\"confirmed\": {}}}}", false, true, "0B63C639AAAAAAAA|BC1 history 6"),
    ("bc1 file without codec marker keeps built-in", Bc1Entry6, 1, 8, "{\"bc1.json\": {\"confirmed\": {\"6\": [1, 1, 1, 1, 1, 1]}}}", true, true, "0B63C639AAAAAAAA|BC1 history 6"),
    ("bc1 folder without bc1.json keeps built-in", Bc1Entry6, 1, 8, "{\"bc4.json\": {\"codec\": \"bc4\", \"confirmed\": {\"25\": [1, 2]}}}", true, true, "0B63C639AAAAAAAA|BC1 history 6"),
    ("bc1 string and float values as Python int()", Bc1Entry6, 1, 8, "{\"bc1.json\": {\"codec\": \"bc1\", \"confirmed\": {\" 6\": [\"7\", 8.9, true, \"+1\", 0, 2]}}}", true, false, "01390208AAAAAAAA"),
    ("bc4 unknown entry, no tables, strict", Bc4Entry1, 4, 8, null, true, false, "refused:BC4 block 0 references unknown built-in history entry 1"),
    ("bc4 unknown entry confirmed by table", Bc4Entry1, 4, 8, "{\"bc4.json\": {\"codec\": \"bc4\", \"confirmed\": {\"1\": [200, 17]}}}", true, false, "C811499224499224"),
    ("bc4 provisional entry confirmed by table", Bc4Entry25, 4, 8, "{\"bc4.json\": {\"codec\": \"bc4\", \"confirmed\": {\"25\": [10, 250]}}}", true, false, "0AFA499224499224"),
    ("bc4 replace without the entry", Bc4Entry25, 4, 8, "{\"bc4.json\": {\"codec\": \"bc4\", \"replace\": true, \"confirmed\": {\"1\": [1, 1]}}}", true, true, "refused:BC4 block 0 references unknown built-in history entry 25"),
};

var malformedCases = new (string Name, string Docs, bool Folder, string Message)[]
{
    ("value out of range", "{\"bc1.json\": {\"codec\": \"bc1\", \"confirmed\": {\"6\": [32, 0, 0, 0, 0, 0]}}}", true, "bc1 confirmed[6] value 32 outside 0..31"),
    ("wrong tuple length", "{\"bc1.json\": {\"codec\": \"bc1\", \"confirmed\": {\"6\": [1, 2, 3]}}}", true, "bc1 confirmed[6] must have 6 integers"),
    ("index out of range", "{\"bc1.json\": {\"codec\": \"bc1\", \"confirmed\": {\"256\": [1, 2, 3, 4, 5, 6]}}}", true, "history entry index 256 outside 0..255"),
    ("list not 256 entries", "{\"bc1.json\": {\"codec\": \"bc1\", \"confirmed\": [[0, 0, 0, 0, 0, 0]]}}", true, "expected 256 entries, got 1"),
    ("table not list or dict", "{\"bc1.json\": {\"codec\": \"bc1\", \"provisional\": 5}}", true, "history table must be a 256-element list or index-keyed dict"),
    ("section not an object", "{\"history.json\": {\"bc1\": [1, 2]}}", false, "bc1 history section must be a JSON object"),
    ("codec mismatch", "{\"history.json\": {\"bc1\": {\"codec\": \"bc4\"}}}", false, "expected codec 'bc1', got 'bc4'"),
    ("negative value", "{\"bc1.json\": {\"codec\": \"bc1\", \"provisional\": {\"6\": [-1, 0, 0, 0, 0, 0]}}}", true, "bc1 provisional[6] value -1 outside 0..31"),
    ("top level not an object", "{\"history.json\": [1, 2, 3]}", false, "<path>: expected a JSON object"),
};

string historyRoot = Directory.CreateTempSubdirectory("bcpack-history-test").FullName;
try
{
    int n = 0;
    foreach (var c in decodeCases)
    {
        BcpackHistory? history = c.Docs is null ? null : BcpackHistory.Load(WriteDocs(Path.Combine(historyRoot, (n++).ToString()), c.Docs, c.Folder));
        string actual;
        try
        {
            var r = BcpackDecoder.DecodeWithInfo(c.Chunk(), c.Codec, c.Size, c.Provisional, history);
            actual = Convert.ToHexString(r.Data) + (r.ProvisionalEntries.Count > 0 ? "|" + string.Join(";", r.ProvisionalEntries.Order(StringComparer.Ordinal)) : "");
        }
        catch (BcpackProvisionalException e) { actual = "refused:" + e.Message.TrimEnd('.'); }
        Check(actual == c.Expected, $"history decode '{c.Name}': expected {c.Expected}, got {actual}");
    }
    foreach (var c in malformedCases)
    {
        string path = WriteDocs(Path.Combine(historyRoot, (n++).ToString()), c.Docs, c.Folder);
        try { BcpackHistory.Load(path); Check(false, $"malformed history '{c.Name}' is refused"); }
        catch (InvalidDataException e) { Check(e.Message.Replace(path, "<path>") == c.Message, $"malformed history '{c.Name}': expected {c.Message}, got {e.Message}"); }
    }
    try { BcpackHistory.Load(Path.Combine(historyRoot, "missing")); Check(false, "a missing history path is refused"); }
    catch (FileNotFoundException) { }
    var bc7Only = BcpackHistory.Load(WriteDocs(Path.Combine(historyRoot, "bc7"), "{\"bc7.json\": {\"historyEndpoints\": {}}}", true));
    Check(bc7Only.Codecs.Count == 0, "a folder with only BC6H/BC7 tables loads no BC1-BC5 table");
    // Without tables (or with a null history) the decode is the built-in one, byte for byte.
    Check(BcpackDecoder.Decode(Bc1Entry6(), 1, 8, true).AsSpan().SequenceEqual(BcpackDecoder.Decode(Bc1Entry6(), 1, 8, true, null)), "null history is the built-in table");
}
finally { Directory.Delete(historyRoot, true); }
Console.WriteLine("BCPack history table tests passed.");

static string WriteDocs(string dir, string docs, bool folder)
{
    Directory.CreateDirectory(dir);
    string? last = null;
    foreach (var p in System.Text.Json.JsonDocument.Parse(docs).RootElement.EnumerateObject())
        File.WriteAllText(last = Path.Combine(dir, p.Name), p.Value.GetRawText());
    return folder ? dir : last!;
}

static byte[] Bc1Entry6() => Convert.FromHexString("10101010101010100203030303030303010120402020402000010410040410040B0000000000000002051105051105100310101001030303010101010001010102000000020202");
static byte[] Bc1Entry255() => Convert.FromHexString("1010101010101010020303030303030301012040202040200001041004041004020000000000000002051105051105100310101001030303010101010001010102000000020202");
static byte[] Bc4Entry1() => Convert.FromHexString("1010101010101010020303030303030301010101404040400001010110101010010000000000000002020211111111100310101040030303104040400010101011000000111111");
static byte[] Bc4Entry25() => Convert.FromHexString("1010101010101010020303030303030301010101404040400001010110101010310000000000000002020211111111100310101040030303104040400010101011000000111111");

static void Check(bool ok, string what)
{
    if (!ok) throw new Exception($"Failed: {what}");
}

static RecoveryIndexCache.CachedIndex Index(string coverage, bool complete) =>
    new("rpfpreview-payload-index/models/4", "models", "levels.rpf", 0, 0, "", coverage, []) { IsComplete = complete };

static void Expect(ReadOnlySpan<byte> system, int graphicsBytes, string expected)
{
    string actual = ModelDecodeFailure.Explain(system, graphicsBytes).Code;
    if (actual != expected) throw new Exception($"Expected {expected}, got {actual}");
}
