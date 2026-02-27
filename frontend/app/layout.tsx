import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "TrueFaceSwapVideo",
  description: "Local-first orchestrator for Runpod serverless jobs",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
