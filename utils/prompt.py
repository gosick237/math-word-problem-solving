import json, re
prompt_template_path = "./prompt/template.json"

def call_prompt_template(prompt_style, inst_style):
    with open(prompt_template_path, 'r') as file:
        data = json.load(file)
        prompt_templates = data["prompt_template"]
        instructions = data["instructions"]

        prompt = prompt_templates[prompt_style][0] + instructions[inst_style] + prompt_templates[prompt_style][1]
        return (
            prompt,
            prompt_templates[prompt_style]
        )

def remove_repeated_sequences(text):
    repeat_num = 3
    # This function finds all substrings that repeat three or more times and are at least one character long.
    for length in range(2, min([len(text) // repeat_num + 1, 10])):  # Maximum possible repeating unit size
        pattern = '(?=(.{%d,}?)(?:\\1{2,}))' % length
        for match in re.finditer(pattern, text):
            sequence = match.group(1)
            if len(sequence) * repeat_num <= len(text):  # To ensure it appears at least three times
                if not '# Answer' in sequence:
                    text = text.replace(sequence, ' ')
    return text

def split_with_pattern(pattern, input_string) :
    split_strings = re.split(pattern, input_string)
    split_strings = [s.strip() for s in split_strings if s.strip()]
    return split_strings

def split_llama2_prompt(input_string):
    inst_comp = []

    if "[/INST]" in input_string:
        inst_comp = split_with_pattern(r"\[/INST\]|\[/INST\]\n\n", input_string)
    else:
        inst_comp = split_with_pattern(r"\n\n", input_string)
    
    # Exception Handing : if More than 2 items.
    if len(inst_comp) > 2:
        out = []
        completion = ''
        for item in inst_comp[1:]:
            completion = completion + item
        out.append(inst_comp[0])
        out.append(completion)
        return out
    
    return inst_comp