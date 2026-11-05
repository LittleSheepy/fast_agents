import json, os, resource, tempfile
from fast_agents.trace import JsonlStore, SpanRecord, SpanKind

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "trace.jsonl")

store = JsonlStore(path)                       # 真·文件落盘模式
print("[1] store._fp is None:", store._fp is None)     # False
print("[2] 初始 store._mem 条数:", len(store._mem))     # 0

payload = "x" * 20000                          # 每条 span 约 40KB
N = 2000
for i in range(N):
    store.write(SpanRecord(
        trace_id="trace-1", span_id=f"span-{i:04d}",
        name="llm.invoke", kind=SpanKind.LLM,
        input={"prompt": payload}, output={"text": payload},
    ))

print("[3] 写入 %d 条后 store._mem 条数: %d" % (N, len(store._mem)))
print("[4] 磁盘文件大小:            %.2f MB" % (os.path.getsize(path)/1024/1024))
mem_bytes = sum(len(json.dumps(d, ensure_ascii=False)) for d in store._mem)
print("[5] store._mem 累积 JSON 量: %.2f MB" % (mem_bytes/1024/1024))
print("[6] 进程峰值 RSS:             %.1f MB" % rss_mb())
store.close()

store2 = JsonlStore(path)                      # 模拟"进程重启/新开 store 指向同一文件"
print("[7] 新建 store 后 _mem 条数:  %d" % len(store2._mem))
print("[8] 第二次进程峰值 RSS:       %.1f MB" % rss_mb())
store2.close()
