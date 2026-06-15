import type { NextConfig } from "next";

// Static export: `next build` emits a fully static site to ./out, which the
// Gradio server (app.py) serves at "/". No Node runtime in the Space.
const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
};

export default nextConfig;
