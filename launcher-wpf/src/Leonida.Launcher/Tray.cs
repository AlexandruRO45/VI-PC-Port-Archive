using System.IO;
using System.Windows;
using WF = System.Windows.Forms;

namespace Leonida.Launcher;

/// <summary>
/// The notification-area icon. It shows while the window is closed to the tray during a build or while the port runs
/// (tooltip: what the build is doing), and for a moment when a Windows notification is shown, which on Windows 10
/// and 11 is a balloon from this icon (no Start-menu shortcut or app identity needed). Lives as long as the app, so a
/// window reload (language, theme) keeps it.
/// </summary>
public sealed class Tray : IDisposable
{
    public static Tray Instance { get; } = new();

    WF.NotifyIcon? icon;
    WF.ToolStripMenuItem? stopItem;
    bool pinned;                       // shown because the window is in the tray, not only for a notification
    System.Windows.Threading.DispatcherTimer? hideTimer;

    public event Action? OpenRequested;
    public event Action? StopRequested;
    public event Action? QuitRequested;

    WF.NotifyIcon Icon
    {
        get
        {
            if (icon != null) return icon;
            var menu = new WF.ContextMenuStrip { ShowImageMargin = false };
            var openItem = new WF.ToolStripMenuItem("", null, (_, _) => OpenRequested?.Invoke()) { Font = new System.Drawing.Font(WF.Control.DefaultFont, System.Drawing.FontStyle.Bold) };
            stopItem = new WF.ToolStripMenuItem("", null, (_, _) => StopRequested?.Invoke());
            var quitItem = new WF.ToolStripMenuItem("", null, (_, _) => QuitRequested?.Invoke());
            menu.Items.AddRange([openItem, stopItem, new WF.ToolStripSeparator(), quitItem]);
            menu.Opening += (_, _) =>
            {
                // Labels are read when the menu opens, so a language change shows at once.
                openItem.Text = Loc.T("Open Leonida Launcher");
                stopItem.Text = Loc.T(AppState.Instance.IsBuilding ? "Stop the build" : "Stop the port");
                stopItem.Enabled = Builds.Instance.Busy;
                quitItem.Text = Loc.T(Builds.Instance.Busy ? "Stop and quit" : "Quit");
            };
            icon = new WF.NotifyIcon { Icon = LoadIcon(), Text = "Leonida Launcher", ContextMenuStrip = menu };
            icon.MouseClick += (_, e) => { if (e.Button == WF.MouseButtons.Left) OpenRequested?.Invoke(); };
            icon.BalloonTipClicked += (_, _) => OpenRequested?.Invoke();
            return icon;
        }
    }

    static System.Drawing.Icon LoadIcon()
    {
        using var s = Application.GetResourceStream(new Uri("pack://application:,,,/Assets/leonida.ico"))!.Stream;
        return new System.Drawing.Icon(s, WF.SystemInformation.SmallIconSize);
    }

    /// <summary>Shows the icon for as long as the window is in the tray.</summary>
    public void Pin()
    {
        pinned = true;
        hideTimer?.Stop();
        Icon.Visible = true;
    }

    public void Unpin()
    {
        pinned = false;
        if (icon != null) icon.Visible = false;
    }

    /// <summary>The tooltip: at most 127 characters, as Windows allows.</summary>
    public void SetStatus(string text)
    {
        if (icon == null) return;
        var t = "Leonida Launcher" + (string.IsNullOrEmpty(text) ? "" : " · " + text);
        icon.Text = t.Length > 127 ? t[..127] : t;
    }

    /// <summary>A Windows notification. Without the tray, the icon shows only long enough for the notification.</summary>
    public void Notify(string title, string text, bool problem)
    {
        var i = Icon;
        i.Visible = true;
        i.ShowBalloonTip(10_000, title, text, problem ? WF.ToolTipIcon.Warning : WF.ToolTipIcon.Info);
        if (pinned) return;
        hideTimer ??= new System.Windows.Threading.DispatcherTimer { Interval = TimeSpan.FromSeconds(12) };
        hideTimer.Tick -= OnHideTimer;
        hideTimer.Tick += OnHideTimer;
        hideTimer.Stop();
        hideTimer.Start();
    }

    void OnHideTimer(object? sender, EventArgs e)
    {
        hideTimer?.Stop();
        if (!pinned && icon != null) icon.Visible = false;
    }

    public void Dispose()
    {
        if (icon == null) return;
        icon.Visible = false;
        icon.Dispose();
        icon = null;
    }
}
