#!/usr/bin/env python3
"""
GLM-4.7 Rust Agent Data Transpiler (Native Tool Format)
"""
import json
import argparse
import random
from pathlib import Path
from typing import Dict

def format_native_tool_call(func_name: str, args_dict: Dict) -> str:
    """Generate GLM-4 native [TOOL_CALLS] format"""
    # CRITICAL FIX: 'arguments' remains a dict, json.dumps handles the wrapping
    payload = [{
        "name": func_name,
        "arguments": args_dict 
    }]
    return f"[TOOL_CALLS] {json.dumps(payload)}"

def format_observation(output: str) -> str:
    return f"<|observation|>
{output}"

def generate_rust_signature(func_def: Dict) -> str:
    name = func_def.get("name", "unknown")
    params = func_def.get("parameters", {}).get("properties", {})
    param_list = [f"{k}: String" for k in params.keys()] # Simplified for robustness
    param_str = ", ".join(param_list)
    return f"pub fn {name}({param_str}) -> Result<String, Error>"

def transpile_sample(sample: Dict) -> Dict:
    conversations = []
    
    # 1. System Prompt (Tool Definitions)
    functions = sample.get("functions", [])
    if functions:
        rust_sigs = [generate_rust_signature(f) for f in functions]
        sys_msg = "You are a Rust code agent. Available tools:\n\n" + "\n".join(rust_sigs)
        conversations.append({"role": "system", "content": sys_msg})
    
    # 2. User Query
    user_txt = sample.get("chat", "").replace("USER:", "").replace("ASSISTANT:", "").strip()
    if user_txt:
        conversations.append({"role": "user", "content": user_txt})
    
    # 3. Assistant Tool Call
    func_call = sample.get("call", sample.get("function_call", {}))
    if func_call:
        f_name = func_call.get("name")
        f_args = func_call.get("arguments", {})
        if isinstance(f_args, str):
            try: f_args = json.loads(f_args)
            except: f_args = {}
            
        tool_str = format_native_tool_call(f_name, f_args)
        conversations.append({"role": "assistant", "content": tool_str})
        
        # 4. Observation & Response
        tool_res = sample.get("result", sample.get("observation", ""))
        if tool_res:
            conversations.append({"role": "observation", "content": format_observation(str(tool_res))})
            final_res = sample.get("response", "Done.")
            conversations.append({"role": "assistant", "content": final_res})
            
    return {"conversations": conversations}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    print(f"🔄 Transpiling {args.input} -> {args.output}")
    processed = 0
    
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            if args.max_samples and processed >= args.max_samples: break
            try:
                sample = json.loads(line)
                converted = transpile_sample(sample)
                if converted["conversations"]:
                    fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    processed += 1
            except Exception: continue
            
    print(f"✅ Transpiled {processed} samples.")

if __name__ == "__main__":
    main()
