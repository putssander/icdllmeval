from .codiformat import CodiFormat
import pandas as pd
import subprocess
import configparser

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
    llmcodes["confidence"] = pd.to_numeric(llmcodes["confidence"], errors='coerce') 
    llmcodes = llmcodes.sort_values(by=['file', 'confidence'], ascending=[True, False])
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






def eval_x(split, path_x):
    
    config = configparser.ConfigParser()
    config.read('./../resources/config.ini')

    # gold
    path_codiesp = config["codiesp"]['data']
    codiformat = CodiFormat(path_codiesp)
    path_x_gold = codiformat.get_path_x_gold(split=split)
    return eval_x_path(path_x_gold, path_x)


def eval_x_path(path_x_gold, path_x):
    
    config = configparser.ConfigParser()
    config.read('./../resources/config.ini')

    # eval
    path_codiesp_eval = config["codiesp"]['eval']
    path_codiesp_eval_script = path_codiesp_eval + "/codiespX_evaluation.py"
    path_to_codes_D_tsv = path_codiesp_eval + "/codiesp_codes/codiesp-D_codes.tsv"
    path_to_codes_P_tsv = path_codiesp_eval + "/codiesp_codes/codiesp-P_codes.tsv"
    
    # Run the command using subprocess
    command = f"python3 {path_codiesp_eval_script} -g {path_x_gold} -p {path_x} -cD {path_to_codes_D_tsv} -cP {path_to_codes_P_tsv}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Print the command's standard output
    print("Standard Output:", result.stdout)
    # Print the command's standard error (if any)
    print("Standard Error:", result.stderr)
    # Return the result object for further processing if needed
    return result


    
def eval_dp(split, path, code_field):
    import os
    import configparser
    import subprocess

    # Read configurations
    config = configparser.ConfigParser()
    config.read('./../resources/config.ini')

    # eval
    path_codiesp_eval = config["codiesp"]['eval']

    # gold
    path_codiesp = config["codiesp"]['data']
    codiformat = CodiFormat(path_codiesp)
    if code_field == codiformat.DIAGNOSTICO:
        path_gold = codiformat.get_path_d_gold(split=split)
        path_to_codes_tsv = path_codiesp_eval + "/codiesp_codes/codiesp-D_codes.tsv"
    elif code_field == codiformat.PROCEDIMIENTO:
        path_gold = codiformat.get_path_p_gold(split=split)
        path_to_codes_tsv = path_codiesp_eval + "/codiesp_codes/codiesp-P_codes.tsv"


    # Paths for results
    # Commands to run
    command_dp = f"python {os.path.join(path_codiesp_eval, 'codiespD_P_evaluation.py')} -g {path_gold} -p {path} -c {path_to_codes_tsv}"
    command_f1 = f"python {os.path.join(path_codiesp_eval, 'comp_f1_diag_proc.py')} -g {path_gold} -p {path} -c {path_to_codes_tsv}"

    # Run the commands using subprocess and capture outputs
    result_dp = subprocess.run(command_dp, shell=True, capture_output=True, text=True)
    result_f1 = subprocess.run(command_f1, shell=True, capture_output=True, text=True)

    # # Write outputs to result files
    # with open(path_results_dp, 'w') as f:
    #     f.write(result_dp.stdout)
    # with open(path_results_f1, 'w') as f:
    #     f.write(result_f1.stdout)

    # Print the outputs
    print("DP Evaluation Results:")
    print(result_dp.stdout)
    print(result_dp.stderr)

    print("F1 Score Results:")
    print(result_f1.stdout)
    print(result_dp.stderr)

    # Return results if necessary
    return result_dp, result_f1
