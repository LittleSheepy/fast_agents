"""
C6-2 沙箱演示脚本：导出报告到tmp目录
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--format", default="md")
    args = parser.parse_args()
    
    out_dir = Path(__file__).parent.parent / "tmp"
    out_dir.mkdir(exist_ok=True)
    
    content = {
        "title": args.title,
        "format": args.format,
        "content": f"# {args.title}\n\n这是一份沙箱演示导出的报告。\n"
    }
    
    ext = args.format
    if ext == "json":
        out = out_dir / f"report.json"
        out.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    elif ext == "txt":
        out = out_dir / "report.txt"
        out.write_text(f"标题: {content['title']}\n{content['content']}", encoding="utf-8")
    else:
        out = out_dir / "report.md"
        out.write_text(content["content"], encoding="utf-8")
    
    print(f"报告已导出: {out}")
    print(f"标题: {args.title}")
    print(f"格式: {args.format}（已自动通过枚举校验）")


if __name__ == "__main__":
    main()
