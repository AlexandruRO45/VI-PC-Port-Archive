using System.ComponentModel;
using System.Runtime.InteropServices;

namespace Leonida.Core;

/// <summary>
/// A Windows job object holding the build and every process it starts (translator shards, llvm-mc, cmake, MSBuild,
/// dotnet, vk_pipeline_check). Closing the handle kills them all (KILL_ON_JOB_CLOSE), so neither Cancel nor a crash
/// of the launcher itself leaves orphaned compilers running; it also gives the tree's CPU time and process list.
/// </summary>
public sealed class JobObject : IDisposable
{
    readonly IntPtr handle;

    public JobObject()
    {
        handle = CreateJobObjectW(IntPtr.Zero, null);
        if (handle == IntPtr.Zero) throw new Win32Exception();
        var info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!SetInformationJobObject(handle, 9 /*ExtendedLimitInformation*/, ref info, (uint)Marshal.SizeOf(info)))
            throw new Win32Exception();
    }

    public void Assign(System.Diagnostics.Process p)
    {
        if (!AssignProcessToJobObject(handle, p.Handle)) throw new Win32Exception();
    }

    public void Terminate(uint code = 1) => TerminateJobObject(handle, code);

    /// <summary>User plus kernel CPU time of every process that ran in the job, and how many still run.</summary>
    public (TimeSpan cpu, int active) Accounting()
    {
        if (!QueryInformationJobObject(handle, 1 /*BasicAccountingInformation*/, out JOBOBJECT_BASIC_ACCOUNTING_INFORMATION a,
                                       (uint)Marshal.SizeOf<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>(), IntPtr.Zero))
            return (TimeSpan.Zero, 0);
        return (TimeSpan.FromTicks(a.TotalUserTime + a.TotalKernelTime), (int)a.ActiveProcesses);
    }

    /// <summary>The ids of the processes running in the job now.</summary>
    public unsafe List<int> ProcessIds()
    {
        const int max = 1024;
        var buffer = new byte[8 + 8 * max];   // two DWORDs, then ULONG_PTR ids
        fixed (byte* b = buffer)
        {
            if (!QueryInformationJobObject(handle, 3 /*BasicProcessIdList*/, (IntPtr)b, (uint)buffer.Length, IntPtr.Zero))
                return [];
            var count = *(uint*)(b + 4);
            var ids = new List<int>((int)count);
            for (var i = 0; i < count && i < max; i++) ids.Add((int)*(ulong*)(b + 8 + 8 * i));
            return ids;
        }
    }

    public void Dispose() => CloseHandle(handle);

    const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000;

    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
        public long PerProcessUserTimeLimit, PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass, SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct IO_COUNTERS { public ulong a, b, c, d, e, f; }

    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_BASIC_ACCOUNTING_INFORMATION
    {
        public long TotalUserTime, TotalKernelTime, ThisPeriodTotalUserTime, ThisPeriodTotalKernelTime;
        public uint TotalPageFaultCount, TotalProcesses, ActiveProcesses, TotalTerminatedProcesses;
    }

    [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern IntPtr CreateJobObjectW(IntPtr attributes, string? name);
    [DllImport("kernel32", SetLastError = true)]
    static extern bool SetInformationJobObject(IntPtr job, int cls, ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION info, uint size);
    [DllImport("kernel32", SetLastError = true)]
    static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32", SetLastError = true)]
    static extern bool TerminateJobObject(IntPtr job, uint code);
    [DllImport("kernel32", SetLastError = true)]
    static extern bool QueryInformationJobObject(IntPtr job, int cls, out JOBOBJECT_BASIC_ACCOUNTING_INFORMATION info, uint size, IntPtr ret);
    [DllImport("kernel32", SetLastError = true)]
    static extern bool QueryInformationJobObject(IntPtr job, int cls, IntPtr info, uint size, IntPtr ret);
    [DllImport("kernel32", SetLastError = true)]
    static extern bool CloseHandle(IntPtr h);
}

/// <summary>
/// Ctrl+C for a build started without a window: it still has a console (CREATE_NO_WINDOW), so the launcher attaches
/// to it for a moment and raises the event there. make_game.py takes that as KeyboardInterrupt, stops its shards and
/// prints "Stopped. Run it again to continue where it left off." exactly as in a terminal.
/// </summary>
public static class ConsoleControl
{
    static readonly object Gate = new();

    public static bool SendCtrlC(int pid)
    {
        lock (Gate)
        {
            FreeConsole();
            if (!AttachConsole((uint)pid)) return false;
            SetConsoleCtrlHandler(IntPtr.Zero, true);   // the launcher itself ignores the event it raises
            var ok = GenerateConsoleCtrlEvent(0 /*CTRL_C_EVENT*/, 0);
            FreeConsole();
            // The event reaches this process too, on a thread of its own a moment later; only after that may Ctrl+C
            // handling come back, or the default handler would end the launcher.
            Thread.Sleep(500);
            SetConsoleCtrlHandler(IntPtr.Zero, false);
            return ok;
        }
    }

    /// <summary>
    /// Called just before a build starts. "Ignore Ctrl+C" is inherited by every process a process starts: a launcher
    /// started from a shell that set it (Git Bash does), or one that just sent a Ctrl+C, would otherwise start builds
    /// that never receive one, and Stop would always have to end them by force.
    /// </summary>
    public static void AllowCtrlCInChildren()
    {
        lock (Gate) SetConsoleCtrlHandler(IntPtr.Zero, false);
    }

    [DllImport("kernel32", SetLastError = true)] static extern bool AttachConsole(uint pid);
    [DllImport("kernel32", SetLastError = true)] static extern bool FreeConsole();
    [DllImport("kernel32", SetLastError = true)] static extern bool SetConsoleCtrlHandler(IntPtr handler, bool add);
    [DllImport("kernel32", SetLastError = true)] static extern bool GenerateConsoleCtrlEvent(uint ev, uint group);
}

public static class SystemMemory
{
    [StructLayout(LayoutKind.Sequential)]
    struct MEMORYSTATUSEX
    {
        public uint dwLength, dwMemoryLoad;
        public ulong ullTotalPhys, ullAvailPhys, ullTotalPageFile, ullAvailPageFile, ullTotalVirtual, ullAvailVirtual,
            ullAvailExtendedVirtual;
    }

    [DllImport("kernel32", SetLastError = true)] static extern bool GlobalMemoryStatusEx(ref MEMORYSTATUSEX m);

    /// <summary>Physical memory Windows can use, and how much of it is free.</summary>
    public static (ulong total, ulong available) Status()
    {
        var m = new MEMORYSTATUSEX { dwLength = (uint)Marshal.SizeOf<MEMORYSTATUSEX>() };
        return GlobalMemoryStatusEx(ref m) ? (m.ullTotalPhys, m.ullAvailPhys) : (0, 0);
    }

    [DllImport("kernel32", SetLastError = true)] static extern bool GetPhysicallyInstalledSystemMemory(out ulong kb);

    /// <summary>The RAM installed in the slots (what the firmware reports), in bytes; 0 if unknown.</summary>
    public static ulong Installed() => GetPhysicallyInstalledSystemMemory(out var kb) ? kb * 1024 : 0;
}
