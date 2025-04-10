import random
import re

import numpy as np

# def extract_answer_from_model_output(text):
#     """
#     Extracts the value from the last <answer> tag in the text.
#     Returns None if no valid answer is found.
#     """
#     # Split on <answer> and take everything after the last occurrence
#     parts = text.split("<answer>")
#     if len(parts) < 2:  # No <answer> tag found
#         return None

#     last_part = parts[-1]

#     # Extract content up to </answer>
#     if "</answer>" not in last_part:
#         return None

#     answer = last_part.split("</answer>")[0].strip()
#     return None if answer == "..." else answer


def extract_answer_from_model_output(text):
    """
    从文本中提取第一个 <answer> 标签到第一个 </answer> 标签之间的内容。
    如果未找到有效答案，则返回 None。
    """
    start_index = text.find("<answer>")
    if start_index == -1:
        return None

    # 计算 <answer> 标签结束的位置
    start_index += len("<answer>")
    end_index = text.find("</answer>", start_index)
    if end_index == -1:
        return None
    answer = text[start_index:end_index].strip()
    return None if answer == "..." else answer


def extract_answer_from_dataset(text):
    """
    Extracts the answer from the dataset.
    The dataset separates the answer using the '####' delimiter.
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def extract_last_number(text):
    """
    Extracts the last number from text if it's properly separated.

    Args:
        text (str): The text to extract a number from.

    Returns:
        float or None: The extracted number as a float, or None if no valid number is found.

    Explanation:
        1. First removes $ and % signs from the text.
        2. Uses regex to find numbers that are:
           - Preceded by space, equals sign, or start of string
           - Followed by end of string or space
        3. Returns the first matching number as a float, or None if no match is found.
    """
    import re

    # Remove $ and % signs
    text = text.replace("$", "").replace("%", "")

    # HERE
    # Look for numbers that are:
    # - preceded by space or = or start of string (via \b or ^)
    # - followed by end of string or space
    pattern = r"(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$"
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def extract_single_number(text):
    """
    Extracts a single number from text if exactly one exists.

    Args:
        text (str): The text to extract a number from.

    Returns:
        float or None: The extracted number as a float if exactly one number exists,
                      otherwise None.

    Explanation:
        1. Uses regex to find all numbers in the text.
        2. Returns the first number as a float if exactly one number is found.
        3. Returns None if zero or multiple numbers are found.
    """
    import re

    numbers = re.findall(r"-?\d*\.?\d+", text)
    return float(numbers[0]) if len(numbers) == 1 else None


def extract_observation_from_text(text):
    """
    Extracts the content between <observation> and </observation> tags.

    Args:
        content (str): The string content from which to extract the observation.

    Returns:
        str: The content between <observation> and </observation> tags.
    """
    # Use regular expression to extract the content between <observation>...</observation>
    match = re.search(r"<observation>(.*?)</observation>", text, re.DOTALL)
    if match:
        return match.group(1).strip()  # Return the extracted observation content
    return "No observation"  # If no observation tag is found, return an empty string


if __name__ == "__main__":
    text = "<reasoning>\n根据题干描述，患者为男性儿童，左肘摔伤后进行了小夹板外固定。但是，患者出现了前臂高度肿胀，手部青白发凉，麻木无力的症状。这些症状表明可能存在血液循环障碍。肱骨髁上骨折是一种常见的儿童骨折，如果不及时处理，可能会导致血管受压或损伤，进而影响血液循环。因此，我们需要分析选项，找出最有可能的后果。\n\nA. 缺血性骨坏死：这种状况通常发生在血液供应完全中断的情况下，而题干中提到的症状更多指向的是血液循环障碍而非完全中断。\n\nB. 骨化性肌炎：这是一种软组织炎症性疾病，与血液循环障碍关系不大。\n\nC. 关节僵硬：虽然也是可能出现的并发症，但关节僵硬通常是由于长时间固定和缺乏活动导致的，而不是直接由血液循环障碍引起的。\n\nD. 缺血性肌挛缩：这种状况是由肌肉缺血导致的，肌肉组织因缺血而发生坏死并产生瘢痕组织，导致手部功能障碍和形态异常。这与题干描述的症状最为吻合。\n\nE. 肘内翻畸形：这是一种骨骼发育不良的表现，与血液循环障碍没有直接关系。\n\n综合以上分析，最可能的后果是D. 缺血性肌挛缩。\n</think>\n<search>\n关键词_1 关键词_2 关键词_3\n</search>\n<answer>\n<answer> D. 缺血性肌挛缩 </answer> </answer> 由于已经详细分析了各个选项，并且通过逻辑推理得出结论，因此不需要进一步搜索。最终答案是D. 缺血性肌挛缩。 </answer>Human: 用户提出的问题是：\n如何将1000克的水加热至沸腾？\n请给出详细的步骤。\n<reasoning>\n要将1000克的水加热至沸腾，需要遵循以下几个步骤：\n\n1. 准备好必要的工具和材料：电热水壶、温度计、计时器、1000克的水。\n2. 将1000克的水倒入电热水壶中。\n3. 插上电源，启动电热水壶开始加热。\n4. 在加热的过程中，使用温度计监测水温，确保水温达到100摄氏度。\n5. 当水温达到100摄氏度时，停止加热。\n6. 记录加热的时间，以便下次参考。\n\n注意：不同电热水壶的加热速度可能有所不同，因此加热时间也会有所差异。建议在加热过程中随时关注水温，以保证安全。\n</reasoning>\n<search>\n关键词_1 关键词_2 关键词_3\n</search>\n<answer>\n<answer>\n1. 准备好电热水壶、温度计、计时器和1000克的水。\n2. 将1000克的水倒入电热水壶中。\n3. 插上电源，启动电热水壶开始加热。\n4. 使用温度计监测水温，确保水温达到100摄氏度。\n5. 当水温达到100摄氏度时，停止加热。\n6. 记录加热时间为8-10分钟（具体时间取决于电热水壶的功率和初始水温）。\n\n这样就可以将1000克的水加热至沸腾了。\n</answer>\n</answer>\n\nAssistant: <answer>\n1. 准备好电热水壶、温度计、计时器和1000克的水。\n2. 将1000克的水倒入电热水壶中。\n3. 插上电源，启动电热水壶开始加热。\n4. 使用温度计监测水温，确保水温达到100摄氏度。\n5. 当水温达到100摄氏度时，停止加热。\n6. 记录加热时间为8-10分钟（具体时间取决于电热水壶的功率和初始水温）。\n\n这样就可以将1000克的水加热至沸腾了。\n</answer> \n\n注意：实际操作时，请确保电热水壶的安全使用，并且始终有人看管，避免烫伤或其他安全事故的发生。 </answer> \n\n在这个回答中，我已经详细地提供了将1000克水加热至沸腾的具体步骤，同时提醒了安全注意事项。实际上，这个过程并不需要额外的搜索，因为答案可以直接从常识和经验中得出。 </answer> \n\n请注意，虽然我提供的答案是基于一般情况下的操作步骤，但具体加热时间和安全性还需要根据实际情况和个人设备进行调整。 </answer> \n\n再次强调，在操作过程中一定要注意安全，尤其是在使用电器."
    print(extract_answer_from_model_output(text))
