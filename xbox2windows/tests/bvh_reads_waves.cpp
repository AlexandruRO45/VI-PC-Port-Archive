#include "bvh_fixture.h"
#include "Produce.h"
#include "ReadValues.h"
#include "WaveValues.h"
#include "Wave64.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

using namespace bvh_fixture;

void check_bvh_reads(ID3D12Device* device) {
    constexpr std::uint64_t base = 0x12345678ffffff80ull;
    constexpr Word extent = 1024, count = 128, epochs = 2;
    constexpr size_t frame_bytes = count*8*sizeof(Word);
    std::vector<Row> queries;
    for (Word i=0; i<count; ++i) {
        Word width = 1+i%4;
        std::uint64_t address = base;
        switch ((i/4)%12) {
        case 0: address += 44; break; // Same address across lanes.
        case 1: address += 124; break; // Vector read crosses a 32-bit carry.
        case 2: address += extent-4*width; break;
        case 3: address += extent-4*width+4; break;
        case 4: address -= 4; break;
        case 5: address += (1ull<<32); break; // Same low bits, different allocation.
        case 6: address += 2; break;
        case 7: address = UINT64_MAX-3; break;
        case 8: address += 128; break;
        case 9: address += 4*((i*17)%241); break;
        case 10: address += 4; break;
        default: address += extent; break;
        }
        queries.push_back({static_cast<Word>(address),static_cast<Word>(address>>32),width,0});
    }
    auto args = upload(device,queries);
    constexpr auto writable = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    auto data = make_buffer(device,extent,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COMMON,writable);
    auto out = make_buffer(device,frame_bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COMMON,writable);
    auto back = make_buffer(device,epochs*frame_bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
    Kernel producer(device,{Produce,sizeof(Produce)}), consumer(device,{ReadValues,sizeof(ReadValues)});
    ComputeCommands cmd(device,producer.pipeline.Get());
    auto* list = cmd.list.Get();
    for (Word epoch=1; epoch<=epochs; ++epoch) {
        transition(list,data.Get(),epoch==1 ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetPipelineState(producer.pipeline.Get());
        list->SetComputeRootSignature(producer.root.Get());
        Row settings{static_cast<Word>(base),static_cast<Word>(base>>32),extent,epoch};
        list->SetComputeRoot32BitConstants(1,4,settings.data(),0);
        list->SetComputeRootUnorderedAccessView(0,data->GetGPUVirtualAddress());
        list->Dispatch(4,1,1);
        transition(list,data.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(list,out.Get(),epoch==1 ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_COPY_SOURCE,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetPipelineState(consumer.pipeline.Get());
        list->SetComputeRootSignature(consumer.root.Get());
        list->SetComputeRootShaderResourceView(0,data->GetGPUVirtualAddress());
        list->SetComputeRootShaderResourceView(1,args->GetGPUVirtualAddress());
        list->SetComputeRootUnorderedAccessView(2,out->GetGPUVirtualAddress());
        list->SetComputeRoot32BitConstants(3,4,settings.data(),0);
        list->Dispatch(count/64,1,1);
        transition(list,out.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(back.Get(),(epoch-1)*frame_bytes,out.Get(),0,frame_bytes);
    }
    cmd.submit_and_wait(device);
    auto result = read(back.Get(),epochs*count*8);
    for (Word epoch=1; epoch<=epochs; ++epoch) for (Word i=0;i<count;++i) {
        auto q = queries[i];
        const auto address = (std::uint64_t{q[1]}<<32) | q[0];
        const bool valid = address>=base && address<=base+extent-4*q[2] && address%4==0;
        const Word* values = result.data()+((epoch-1)*count+i)*8;
        equal(values[4],valid,i,4);
        const Word limit = epoch==1 ? 37 : 97;
        auto source_word = [limit,epoch](Word index) {
            return index==3 ? limit : (index*0x9e3779b9u) ^ (epoch*0x51f3a19du);
        };
        const Word uniformIndex = 11+(i/64)*64;
        equal(values[5],source_word(uniformIndex),i,5);
        equal(values[6],limit,i,6);
        equal(values[7],i<limit ? source_word(i) : 0xdeadbeefu,i,7);
        for (Word k=0;k<4;++k) {
            Word expected = 0xdeadbeefu;
            if (valid && k<q[2]) expected = source_word(static_cast<Word>((address-base)/4+k));
            equal(values[k],expected,i,k);
        }
    }
    std::printf("reads=%u epochs=%u widths=1,2,3,4 mismatches=0\n",count*epochs,epochs);
}

bool check_bvh_waves(ID3D12Device* device, bool force64) {
    if (force64) {
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 caps{};
        D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
#if defined(OFFSCREEN_AGILITY_620) && defined(__ID3D12Device16_INTERFACE_DEFINED__)
        ComPtr<ID3D12Device16> query;
        check(device->QueryInterface(IID_PPV_ARGS(&query)), "Get Agility wave capabilities");
        check(query->CheckFeatureSupport1(D3D12_FEATURE_D3D12_OPTIONS1,&caps,sizeof(caps),620),"Query wave range");
        check(query->CheckFeatureSupport1(D3D12_FEATURE_SHADER_MODEL,&model,sizeof(model),620),"Query wave shader model");
#else
        check(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1,&caps,sizeof(caps)),"Query wave range");
        check(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL,&model,sizeof(model)),"Query wave shader model");
#endif
        std::printf("native_lane_range=%u..%u shader_model_capped_at_6_6=0x%x\n",
                    caps.WaveLaneCountMin,caps.WaveLaneCountMax,model.HighestShaderModel);
        if (!caps.WaveOps || model.HighestShaderModel<D3D_SHADER_MODEL_6_6 ||
                caps.WaveLaneCountMin>64 || caps.WaveLaneCountMax<64) return false;
    }
    constexpr Word groups=6+32, count=groups*64, fields=32;
    std::vector<Row> queries;
    for (Word i=0;i<count;++i) {
        Word g=i/64, lane=i%64;
        bool vote = g==1 || (g==2 && lane%3==1) || (g==3 && (lane==31 || lane==32)) ||
                    (g==4 && lane==63) || (g==5 && lane%2==0);
        Word value = 0xf0000001u ^ (i*0x319dbu);
        if (g>=6) {
            // Probe every input of both half-wave scans, with distinguishable halves.
            vote = (lane%32 == g-6);
            value = vote ? (lane<32 ? 1u : 17u) : 0u;
        }
        queries.push_back({value,vote,0,0});
    }
    auto args=upload(device,queries);
    constexpr UINT bytes=count*fields*sizeof(Word);
    auto out=make_buffer(device,bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto back=make_buffer(device,bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
    Kernel shader(device,force64 ? D3D12_SHADER_BYTECODE{Wave64,sizeof(Wave64)} :
                                   D3D12_SHADER_BYTECODE{WaveValues,sizeof(WaveValues)});
    ComputeCommands cmd(device,shader.pipeline.Get());
    auto* list=cmd.list.Get();
    transition(list,out.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->SetComputeRootSignature(shader.root.Get());
    list->SetComputeRootShaderResourceView(1,args->GetGPUVirtualAddress());
    list->SetComputeRootUnorderedAccessView(2,out->GetGPUVirtualAddress());
    list->Dispatch(groups,1,1);
    transition(list,out.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(back.Get(),0,out.Get(),0,bytes);
    cmd.submit_and_wait(device);
    auto result=read(back.Get(),count*fields);
    const Word width=result[9];
    if (!width || width>128) throw std::runtime_error("Unexpected wave width");
    if (force64) equal(width,64,0,9);
    for (Word g=0;g<groups;++g) {
        std::map<Word,std::vector<Word>> waves;
        const Row* q=queries.data()+64*g;
        const Word* records=result.data()+64*g*fields;
        Row logical_mask{};
        for (Word t=0;t<64;++t) logical_mask[t/32] |= q[t][1]<<(t%32);
        Word rank=0, fullPrefix=0;
        for (Word t=0;t<64;++t) {
            auto r=records+t*fields;
            Word sum=0,total=0;
            for (Word j=t&32;j<(t&32)+32;++j) { total+=q[j][0]; if (j<t) sum+=q[j][0]; }
            Word expected[]{sum+q[t][0],sum,total,q[(7*t+3)%64][0],logical_mask[0],logical_mask[1],rank,q[0][0]};
            for (Word j=0;j<8;++j) equal(r[j],expected[j],64*g+t,j);
            fullPrefix += q[t][0];
            equal(r[28],q[63-t][0],64*g+t,28);
            equal(r[29],q[t^31][0],64*g+t,29);
            equal(r[31],fullPrefix,64*g+t,31);
            if (g==3 && (t==31 || t==32) && r[28]==r[29])
                throw std::runtime_error("Half-local mirror missed its boundary witness");
            rank+=q[t][1];
            if (r[8]>=width || r[10]>=64) throw std::runtime_error("Invalid native lane record");
            equal(r[9],width,t,9);
            waves[r[10]].push_back(t);
        }
        for (auto& [first,threads]:waves) {
            auto lane=[&](Word t){return records[t*fields+8];};
            std::sort(threads.begin(),threads.end(),[&](Word a,Word b){return lane(a)<lane(b);});
            equal(first,threads.front(),first,10);
            Row mask{};
            Word minimum=UINT32_MAX, active_min=UINT32_MAX, active_first=0, active_count=0;
            for (size_t j=0;j<threads.size();++j) {
                Word t=threads[j], l=lane(t);
                if (j && lane(threads[j-1])==l) throw std::runtime_error("Duplicate native lane");
                minimum=(std::min)(minimum,q[t][0]);
                if (q[t][1]) {
                    mask[l/32]|=1u<<(l%32);
                    if (!active_count) active_first=q[t][0];
                    ++active_count;
                    active_min=(std::min)(active_min,q[t][0]);
                }
            }
            Word prefix=0,sum=0;
            for (Word t:threads) {
                auto r=records+t*fields;
                const Word mirror = width==64 && threads.size()==64 ? q[threads[63-lane(t)]][0] : UINT32_MAX;
                equal(r[30],mirror,t,30);
                equal(r[11],q[threads.back()][0],t,11);
                for (Word j=0;j<4;++j) equal(r[12+j],mask[j],t,12+j);
                Word expected[]{prefix,minimum,sum,q[first][0],prefix,active_min,active_first,active_count};
                for (Word j=0;j<8;++j) equal(r[16+j],j>=4 && !q[t][1] ? UINT32_MAX : expected[j],t,16+j);
                if (threads.size()==width) {
                    Word n=lane(t), scan=0;
                    for (Word j=n & ~31u;j<=n;++j) scan+=q[threads[j]][0];
                    Word route[]{scan,q[threads[(n*7+3)%width]][0],width,1};
                    for (Word j=0;j<4;++j) equal(r[24+j],route[j],t,24+j);
                } else {
                    for (Word j=24;j<28;++j) equal(r[j],UINT32_MAX,t,j);
                }
                prefix+=q[t][1]; sum+=q[t][0];
            }
        }
    }
    std::printf("native_wave_width=%u logical_width=64 groups=%u fields_per_thread=%u mismatches=0\n",width,groups,fields);
    return true;
}
