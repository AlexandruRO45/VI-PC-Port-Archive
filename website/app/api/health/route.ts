export function GET() {
  return Response.json({ ok: true, service: "leonida-project-hub" }, {
    headers: { "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" },
  });
}
