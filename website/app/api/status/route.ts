import { PROJECT_STATUS } from "../../lib/project-data";

export function GET() {
  return Response.json(PROJECT_STATUS, {
    headers: {
      "Cache-Control": "public, max-age=300",
      "X-Content-Type-Options": "nosniff",
    },
  });
}
