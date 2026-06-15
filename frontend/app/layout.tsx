import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Smol Signals — finance YouTuber track records",
  description:
    "A small model extracts buy/sell/hold calls from finance YouTube transcripts and scores them against SPY.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
