import re
import humps


def get_nlg_service_name(service_name: str) -> str:
    domain, num = service_name.split("_")
    decamelized_domain = humps.decamelize(domain)
    spaced_domain = remove_underscore(decamelized_domain)
    return spaced_domain


def remove_underscore(item: str):
    return item.replace("_", " ")


def get_apicall_method_from_text(text: str, reg_exp=r"method='([^']+)'") -> str:
    try:
        match = re.search(reg_exp, text).group(1)
    except (AttributeError, TypeError):
        match = ""
    return match


def get_parameters_from_text(text: str) -> str:
    match = re.search(r"parameters\s*=\s*\{([^}]+)\}", text)
    if match:
        params_str = match.group(1)
        # Adjusted regex to handle quoted keys and values
        pairs = re.findall(r"(['\"])(\w+)\1\s*:\s*(['\"])(.*?)\3", params_str)
        params = {}
        for match in pairs:
            key = match[1]
            value = match[3]
            params[key] = value
        return params
    else:
        return {}

def get_parameters_from_gpt_generated_text_gpt(text: str) -> str:
    """
    Extract parameters from a model-produced API call.

    Historically, the system model produced a "legacy" style:
        parameters={ location: Sydney, departure_date: 2019-03-04 }

    After adding JSON / output enforcing, many runs now produce a Python-dict style:
        parameters={'location': 'Sydney', 'departure_date': '2019-03-04'}

    The evaluator should accept both, otherwise key/value metrics collapse to ~0.
    """
    # First try the strict quoted format (same as ground truth extraction).
    params = get_parameters_from_text(text)
    if params:
        return params

    # Fall back to legacy unquoted "key: value" pairs.
    match = re.search(r"parameters\s*=\s*\{([^}]+)\}", text)
    if not match:
        return {}

    params_str = match.group(1)
    pairs = re.findall(r"(\w+)\s*:\s*([^,}]+)", params_str)
    out = {}
    for key, value in pairs:
        value = value.strip().strip("'\"")
        out[key.strip()] = value
    return out
