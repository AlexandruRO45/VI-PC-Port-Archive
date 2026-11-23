import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://leonida-heperkoo.meshedgoby9.chatgpt.site"),
  title: "Leonida · Heperkoo | Native PC Research",
  description:
    "Heperkoo's experimental Leonida fork: native GTA VI PC runtime research, documented blockers, GitHub activity, and contributor handoffs. No verified rendered gameplay yet.",
  openGraph: {
    title: "Leonida · Heperkoo — Toward the first real frame",
    description:
      "An independent experimental fork for native PC runtime research, reproducible testing, and collaboration. Follow the work on GitHub.",
    type: "website",
    url: "https://leonida-heperkoo.meshedgoby9.chatgpt.site",
    siteName: "Leonida · Heperkoo",
  },
  twitter: {
    card: "summary",
    title: "Leonida · Heperkoo",
    description: "Native PC runtime research. Documented progress. A reproducible path toward the first real frame.",
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
