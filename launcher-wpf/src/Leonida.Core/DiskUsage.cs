namespace Leonida.Core;

/// <summary>One folder on the Disk space page: what it is, how big, and whether the launcher may delete it.</summary>
public sealed record DiskEntry(string Title, string Description, string Path, bool Deletable, string? Warning)
{
    public long Bytes { get; set; }
    public long Files { get; set; }
}

/// <summary>
/// What a build has written, folder by folder: the work folder's parts (each one a stage makes again when it is
/// gone), the build logs and the launcher's own cache. The port's own files and your game copy are listed only to
/// show their size, never deleted.
/// </summary>
public static class DiskUsage
{
    /// <summary>The work folder's parts that make_game.py writes, with what deleting each one costs.</summary>
    static readonly (string Rel, string Title, string Description, string? Warning)[] Known =
    [
        (@"shaders\fspx", "Unpacked shader packages", "The shader packages taken out of your copy. The shaders stage unpacks them again.", null),
        (@"shaders\dxbc", "Disassembled shaders", "The shaders as text, the translator's input. The shaders stage makes them again.", null),
        (@"shaders\spirv", "Translated shaders", "The SPIR-V the port runs. PLAY needs them; without them the shaders stage translates everything again (the longest step).",
            "PLAY needs these. The shaders stage will translate every shader again, which takes the longest."),
        ("assets", "Exported assets", "Models, textures and audio from the assets stage. Only needed to look at the assets; the assets stage exports them again.",
            "The assets stage takes hours to export these again."),
        ("tools", "Built tools", "Tools the build compiles for itself (rpf9dump). The next build compiles them again.", null),
    ];

    /// <summary>The folders to show for these locations. Sizes are 0 until <see cref="Measure"/>.</summary>
    public static List<DiskEntry> Entries(string outFolder, string workFolder, string logs, string launcherCache)
    {
        var list = new List<DiskEntry>();
        var shown = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var (rel, title, description, warning) in Known)
        {
            var p = Path.Combine(workFolder, rel);
            if (!Directory.Exists(p)) continue;
            list.Add(new DiskEntry(title, description, p, true, warning));
            shown.Add(Path.GetFullPath(p));
        }
        // Anything else in the work folder (and in its shaders folder): shown under its own name.
        foreach (var parent in new[] { workFolder, Path.Combine(workFolder, "shaders") })
        {
            if (!Directory.Exists(parent)) continue;
            foreach (var d in SafeDirectories(parent))
            {
                var full = Path.GetFullPath(d);
                if (shown.Contains(full) || full.Equals(Path.GetFullPath(Path.Combine(workFolder, "shaders")), StringComparison.OrdinalIgnoreCase)) continue;
                var rel = Path.GetRelativePath(workFolder, full);
                list.Add(new DiskEntry(rel, "Other work files of the build. A stage makes them again when it needs them.", full, true, null));
                shown.Add(full);
            }
        }
        if (Directory.Exists(outFolder))
        {
            var port = new DiskEntry("The port", "gta6rt.exe and the files next to it. Delete it with a new build or by hand.", outFolder, false, null);
            list.Insert(0, port);
        }
        if (Directory.Exists(logs))
            list.Add(new DiskEntry("Build logs", "The full output of every build and run.", logs, true, null));
        if (Directory.Exists(launcherCache))
            list.Add(new DiskEntry("Launcher cache", "The Can I Run page's browser data and the Credits list. Made again when needed.", launcherCache, true, null));
        return list;
    }

    static IEnumerable<string> SafeDirectories(string parent)
    {
        try { return Directory.EnumerateDirectories(parent).Where(d => !IsLink(d)).ToList(); }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { return []; }
    }

    static bool IsLink(string path)
    {
        try { return new DirectoryInfo(path).Attributes.HasFlag(FileAttributes.ReparsePoint); }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { return true; }
    }

    /// <summary>Bytes and files under a folder; links (junctions, symlinks) are not followed. <paramref name="skip"/>
    /// leaves out a folder inside it (the work folder inside the output folder, counted on its own).</summary>
    public static (long Bytes, long Files) Measure(string folder, string? skip = null, CancellationToken cancel = default)
    {
        long bytes = 0, files = 0;
        var pending = new Stack<string>();
        pending.Push(folder);
        var skipFull = skip == null ? null : Path.GetFullPath(skip).TrimEnd('\\');
        while (pending.Count > 0)
        {
            cancel.ThrowIfCancellationRequested();
            var dir = pending.Pop();
            try
            {
                foreach (var f in new DirectoryInfo(dir).EnumerateFileSystemInfos())
                {
                    if (f.Attributes.HasFlag(FileAttributes.ReparsePoint)) continue;
                    if (f is DirectoryInfo d)
                    {
                        if (skipFull != null && d.FullName.TrimEnd('\\').Equals(skipFull, StringComparison.OrdinalIgnoreCase)) continue;
                        pending.Push(d.FullName);
                    }
                    else if (f is FileInfo fi) { bytes += fi.Length; files++; }
                }
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
        }
        return (bytes, files);
    }

    /// <summary>
    /// Why the launcher won't delete <paramref name="path"/>, or null when it may: it must be a folder strictly inside
    /// one of <paramref name="roots"/> (the work folder, the logs, the launcher cache), not a link, and never the game
    /// copy, a folder holding it, or a folder inside it.
    /// </summary>
    public static string? RefuseDelete(string path, IEnumerable<string> roots, string? gameFolder)
    {
        string Full(string p) => Path.GetFullPath(p).TrimEnd('\\', '/');
        var p = Full(path);
        if (!Directory.Exists(p)) return "the folder is gone";
        if (IsLink(p)) return "it is a link to another folder";
        bool Inside(string child, string parent) => child.StartsWith(parent + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase);
        var rootsFull = roots.Where(r => !string.IsNullOrWhiteSpace(r)).Select(Full).ToList();
        if (!rootsFull.Any(r => p.Equals(r, StringComparison.OrdinalIgnoreCase) || Inside(p, r)))
            return "it is not in the work folder, the logs or the launcher's cache";
        if (!string.IsNullOrWhiteSpace(gameFolder))
        {
            var g = Full(gameFolder);
            if (p.Equals(g, StringComparison.OrdinalIgnoreCase) || Inside(g, p) || Inside(p, g)) return "it is (or holds) your game copy";
        }
        if (File.Exists(Path.Combine(p, "MicrosoftGame.Config"))) return "it looks like a game package (MicrosoftGame.Config)";
        return null;
    }
}
