cbuffer Params : register(b0) { uint4 data[4096]; }
RWStructuredBuffer<uint> result : register(u0);
[numthreads(1,1,1)]
void main() {
    result[1] = data[0].x;
    result[2] = data[15].w;
    result[3] = data[2048].y;
    result[4] = data[4095].w;
}
