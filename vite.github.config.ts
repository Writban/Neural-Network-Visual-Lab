import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

function githubPagesBase() {
  const repository = process.env.GITHUB_REPOSITORY;
  if (!repository) return "/";

  const [owner, name] = repository.split("/");
  return name.toLowerCase() === `${owner}.github.io`.toLowerCase()
    ? "/"
    : `/${name}/`;
}

export default defineConfig({
  base: githubPagesBase(),
  plugins: [react()],
  build: {
    outDir: "dist-github",
    emptyOutDir: true,
  },
});
