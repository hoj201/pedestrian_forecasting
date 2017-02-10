import json
def read_json(fname):
    with open(fname) as f:
        st = f.read()
    json_acceptable_string = st.replace("'", "\"")
    dic = json.loads(json_acceptable_string)
    return dic
