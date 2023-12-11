import json
import langchain
def chunk_list(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]

def clean_md(txt):
    if "```json" in txt:
        return langchain.output_parsers.json.parse_json_markdown(txt)
    else:
        return json.loads(txt)

