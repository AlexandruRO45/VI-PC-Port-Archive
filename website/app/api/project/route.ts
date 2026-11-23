import { getProject } from "../../lib/project-data";

export const dynamic = "force-dynamic";

export async function GET() {
  const project = await getProject();
  return Response.json(project, {
    headers: {
      "Cache-Control": "public, max-age=30, s-maxage=60",
      "X-Content-Type-Options": "nosniff",
    },
  });
}
