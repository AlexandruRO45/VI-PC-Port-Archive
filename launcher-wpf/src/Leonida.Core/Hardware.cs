using System.Runtime.InteropServices;
using Microsoft.Win32;

namespace Leonida.Core;

public sealed record GpuInfo(string Name, ulong MemoryBytes, string? Driver, bool Integrated);

public sealed record HardwareInfo(
    string Cpu, int Cores, int Threads, List<GpuInfo> Gpus, ulong RamInstalledBytes, ulong RamUsableBytes,
    string? RamType, uint? RamSpeedMts, int RamModules, string Os, List<string> Notes)
{
    /// <summary>The GPU a game would run on: the one with the most dedicated memory (a discrete card over an iGPU).</summary>
    public GpuInfo? PrimaryGpu => Gpus.OrderByDescending(g => g.Integrated ? 0 : 1).ThenByDescending(g => g.MemoryBytes).FirstOrDefault();

    public double RamGb => Math.Round((RamInstalledBytes > 0 ? RamInstalledBytes : RamUsableBytes) / 1073741824.0, 1);

    /// <summary>"DDR4" or "DDR5" for the checker's RAM type field: LPDDR counts as its generation; anything older
    /// or unknown is null and the user picks.</summary>
    public string? CheckerRamType => RamType switch
    {
        "DDR5" or "LPDDR5" or "LPDDR5X" => "ddr5",
        "DDR4" or "LPDDR4" or "LPDDR4X" => "ddr4",
        _ => null,
    };
}

/// <summary>
/// The PC's CPU, GPUs and memory, read locally from Windows (registry, SMBIOS through WMI, the memory API). Nothing
/// is sent anywhere by this class.
/// </summary>
public static class HardwareDetector
{
    const string DisplayClass = @"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}";

    public static HardwareInfo Detect()
    {
        var notes = new List<string>();
        var cpu = CpuName() ?? "Unknown CPU";
        var (cores, threads) = CoreCount();
        var gpus = Gpus(notes);
        var (total, _) = SystemMemory.Status();
        var installed = SystemMemory.Installed();
        var (type, speed, modules) = MemoryModules(notes);
        return new HardwareInfo(cpu, cores, threads, gpus, installed, total, type, speed, modules, OsName(), notes);
    }

    static string? CpuName()
    {
        using var k = Registry.LocalMachine.OpenSubKey(@"HARDWARE\DESCRIPTION\System\CentralProcessor\0");
        return (k?.GetValue("ProcessorNameString") as string)?.Trim();
    }

    static (int cores, int threads) CoreCount()
    {
        uint len = 0;
        GetLogicalProcessorInformationEx(0 /*RelationProcessorCore*/, IntPtr.Zero, ref len);
        if (len == 0) return (0, Environment.ProcessorCount);
        var buffer = Marshal.AllocHGlobal((int)len);
        try
        {
            if (!GetLogicalProcessorInformationEx(0, buffer, ref len)) return (0, Environment.ProcessorCount);
            int cores = 0;
            for (long offset = 0; offset < len;)
            {
                var size = Marshal.ReadInt32(buffer, (int)offset + 4);   // Relationship, then Size
                cores++;
                offset += size;
            }
            return (cores, Environment.ProcessorCount);
        }
        finally { Marshal.FreeHGlobal(buffer); }
    }

    /// <summary>Each display adapter under the display class key, with its real memory size
    /// (HardwareInformation.qwMemorySize; WMI's AdapterRAM stops at 4 GB). Virtual and remote adapters are left out.</summary>
    static List<GpuInfo> Gpus(List<string> notes)
    {
        var list = new List<GpuInfo>();
        using var cls = Registry.LocalMachine.OpenSubKey(DisplayClass);
        if (cls == null) { notes.Add("The display adapter list could not be read."); return list; }
        foreach (var sub in cls.GetSubKeyNames().Where(n => n.Length == 4 && n.All(char.IsDigit)))
        {
            using var k = cls.OpenSubKey(sub);
            var name = (k?.GetValue("DriverDesc") as string)?.Trim();
            if (string.IsNullOrEmpty(name) || IsVirtual(name)) continue;
            ulong mem = k!.GetValue("HardwareInformation.qwMemorySize") switch
            {
                long l => (ulong)l,
                byte[] b when b.Length >= 8 => BitConverter.ToUInt64(b, 0),
                byte[] b when b.Length >= 4 => BitConverter.ToUInt32(b, 0),
                int i => (uint)i,
                _ => 0,
            };
            if (mem == 0 && k.GetValue("HardwareInformation.MemorySize") is byte[] m && m.Length >= 4) mem = BitConverter.ToUInt32(m, 0);
            if (mem == 0 && k.GetValue("HardwareInformation.MemorySize") is int mi) mem = (uint)mi;
            var driver = k.GetValue("DriverVersion") as string;
            if (list.Any(g => g.Name == name && g.MemoryBytes == mem)) continue;   // one card, several driver keys
            list.Add(new GpuInfo(name, mem, driver, IsIntegrated(name, mem)));
        }
        if (list.Count == 0) notes.Add("No display adapter was found in the registry.");
        return list;
    }

    static bool IsVirtual(string name) =>
        name.Contains("Basic Display", StringComparison.OrdinalIgnoreCase) ||
        name.Contains("Basic Render", StringComparison.OrdinalIgnoreCase) ||
        name.Contains("Remote", StringComparison.OrdinalIgnoreCase) ||
        name.Contains("Virtual", StringComparison.OrdinalIgnoreCase) ||
        name.Contains("Parsec", StringComparison.OrdinalIgnoreCase) ||
        name.Contains("Hyper-V", StringComparison.OrdinalIgnoreCase);

    static bool IsIntegrated(string name, ulong mem)
    {
        var n = name.ToLowerInvariant();
        if (n.Contains("uhd graphics") || n.Contains("iris") || n.Contains("hd graphics") || n.Contains("intel(r) graphics") ||
            n.Contains("arc(tm) graphics") || n.Contains("arc graphics")) return true;
        if (n == "amd radeon(tm) graphics" || n == "amd radeon graphics" || System.Text.RegularExpressions.Regex.IsMatch(n, @"radeon\s*\(tm\)?\s*\d{3}m\b|radeon \d{3}m\b|radeon 8\d{2}0s\b|vega \d+"))
            return true;
        return mem > 0 && mem <= 512UL << 20;
    }

    /// <summary>Memory type and speed from the firmware's SMBIOS memory-device records (type 17), read with
    /// GetSystemFirmwareTable: the data WMI's Win32_PhysicalMemory reports, without WMI. Unknown when the firmware
    /// leaves the fields empty.</summary>
    static (string? type, uint? speed, int modules) MemoryModules(List<string> notes)
    {
        const uint RSMB = 0x52534D42;
        var size = GetSystemFirmwareTable(RSMB, 0, IntPtr.Zero, 0);
        if (size == 0) { notes.Add("The firmware's memory table could not be read; choose the RAM type below."); return (null, null, 0); }
        var buffer = new byte[size];
        unsafe { fixed (byte* b = buffer) GetSystemFirmwareTable(RSMB, 0, (IntPtr)b, size); }
        string? type = null;
        uint? speed = null;
        var modules = 0;
        var len = (int)Math.Min(BitConverter.ToUInt32(buffer, 4), (uint)buffer.Length - 8);
        var i = 8;   // RawSMBIOSData: four version bytes and a DWORD length, then the structures
        while (i + 4 <= 8 + len)
        {
            int kind = buffer[i], formatted = buffer[i + 1];
            if (formatted < 4 || i + formatted > buffer.Length) break;
            if (kind == 127) break;   // end of table
            if (kind == 17 && formatted >= 0x17)
            {
                var sizeField = BitConverter.ToUInt16(buffer, i + 0x0C);
                if (sizeField != 0)   // 0: an empty slot
                {
                    modules++;
                    type ??= MemoryTypeName(buffer[i + 0x12]);
                    uint s = BitConverter.ToUInt16(buffer, i + 0x15);
                    if (formatted >= 0x22) { var configured = BitConverter.ToUInt16(buffer, i + 0x20); if (configured is > 0 and < 0xFFFF) s = configured; }
                    if (s is > 0 and < 0xFFFF) speed = speed is uint cur ? Math.Max(cur, s) : s;
                }
            }
            // Skip the formatted area, then the string set, which ends with two zero bytes.
            var j = i + formatted;
            while (j + 1 < buffer.Length && !(buffer[j] == 0 && buffer[j + 1] == 0)) j++;
            i = j + 2;
        }
        if (modules > 0 && type == null) notes.Add("This PC's firmware does not report the memory type; choose it below.");
        return (type, speed, modules);
    }

    [DllImport("kernel32", SetLastError = true)]
    static extern uint GetSystemFirmwareTable(uint provider, uint id, IntPtr buffer, uint size);

    static uint? ToUInt(object? v) => v switch
    {
        null => null,
        DBNull => null,
        _ => uint.TryParse(Convert.ToString(v, System.Globalization.CultureInfo.InvariantCulture), out var u) ? u : null,
    };

    /// <summary>SMBIOS 3.x memory type codes (DSP0134 table 76).</summary>
    public static string? MemoryTypeName(uint? code) => code switch
    {
        18 => "DDR",
        19 => "DDR2",
        24 => "DDR3",
        26 => "DDR4",
        27 => "LPDDR",
        28 => "LPDDR2",
        29 => "LPDDR3",
        30 => "LPDDR4",
        34 => "DDR5",
        35 => "LPDDR5",
        _ => null,
    };

    static string OsName()
    {
        using var k = Registry.LocalMachine.OpenSubKey(@"SOFTWARE\Microsoft\Windows NT\CurrentVersion");
        var build = k?.GetValue("CurrentBuildNumber") as string;
        var product = k?.GetValue("ProductName") as string ?? "Windows";
        if (int.TryParse(build, out var b) && b >= 22000) product = product.Replace("Windows 10", "Windows 11");
        var display = k?.GetValue("DisplayVersion") as string;
        return $"{product}{(display != null ? " " + display : "")} (build {build}), {RuntimeInformation.OSArchitecture}";
    }

    [DllImport("kernel32", SetLastError = true)]
    static extern bool GetLogicalProcessorInformationEx(int relationship, IntPtr buffer, ref uint length);
}
