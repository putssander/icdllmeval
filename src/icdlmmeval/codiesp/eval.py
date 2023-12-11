from .codiformat import CodiFormat

def get_dfs_x_eval(split, llmcodes, code_field):
    print('eval x')
    codiformat = CodiFormat()
    files = llmcodes["file"].unique()
    types = llmcodes["type"].unique()

    df_x = codiformat.get_df_x(split)
    df_x = df_x[df_x["FILE"].isin(files)]
    df_x = df_x[df_x["TYPE"].isin(types)]

    llmcodes_x = llmcodes[['file', 'offsets', 'type', code_field]]
    llmcodes_x.rename(columns={'file': 'FILE', 'offsets': 'OFFSETS', 'type': 'TYPE', code_field: 'CODE'}, inplace=True)

    return df_x, llmcodes_x


def get_dfs_d_eval(split, llmcodes, code_field):
    codiformat = CodiFormat()
    df_gold = codiformat.get_df_d(split)
    return get_dfs_d_p_eval(df_gold, llmcodes, codiformat.DIAGNOSTICO, code_field)

def get_dfs_p_eval(split, llmcodes, code_field):
    codiformat = CodiFormat()
    df_gold = codiformat.get_df_p(split)
    return get_dfs_d_p_eval(df_gold, llmcodes, codiformat.PROCEDIMIENTO, code_field)

def get_dfs_d_p_eval(df_gold, llmcodes, type, code_field):
    print(f'eval type={type}')
    files = llmcodes["file"].unique()
    df_gold = df_gold[df_gold["FILE"].isin(files)]
    llmcodes = llmcodes[llmcodes["type"].isin([type])]
    llmcodes = llmcodes.sort_values(by='confidence', ascending=False)
    llmcodes = llmcodes.drop_duplicates(subset=["file", code_field], keep="first")
    llmcodes = llmcodes[['file', code_field]]
    llmcodes.rename(columns={'file': 'FILE', code_field: 'CODE'}, inplace=True)
    return df_gold, llmcodes


def is_match_parent(code, selected_code, type):
    if type == CodiFormat.DIAGNOSTICO:
        if code[:3].upper() in selected_code[:3].upper():
            return True
        else:
            return False
    else:
        if code[:4].upper() in selected_code[:4].upper():
            return True
        else:
            return False

