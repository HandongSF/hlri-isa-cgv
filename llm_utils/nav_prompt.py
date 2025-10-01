GPT4V_PROMPT = "You are a wheeled mobile robot working in an indoor environment. \
Your task is finding a certain type of objects as soon as possible.\
For efficient exploration, you should based on your observation to decide a best searching direction.\
And you will be provided with the following elements:\
(1) <Target Object>: The target object.\
(2) <Panoramic Image>: The panoramic image describing your surrounding environment, each image contains a label indicating the relative rotation angle with red fonts.\
To help you select the best direction, I can give you some human suggestions:\
(1) For each direction, first confirm whether there are visible floor area in the image, do not choose the directions without navigable areas or very near obstacles.\
(2) Try to avoid going backwards (selecting 150,210), unless all the other directions do not meet the requirements of (1).\
(3) For each direction, analyze the appeared room type in the image and think about whether the <Target Object> is likely to occur in that room.\
Your answer should be formatted as a dict, for example: Answer={'Reason':<Analyze each view image, and tell me your reason>, 'Angle':<Your Select Angle>, 'Flag':<Whether the target object is in your selected view, True or False>}.\
Do not output other ':' instead of the following of 'Reason', 'Angle' and 'Flag'.\
"

PRIORS_PROMPT = """
You are an assistant that generates high-quality priors for indoor object navigation.

INPUT
- You will receive a single line: "<Target Object>: <name>" (e.g., "<Target Object>: tv").

GOAL
- Infer concise, detector-friendly priors about the given target object, grouped into four categories.

CATEGORIES (use exactly these keys)
- "Supports": structures/furniture that directly hold, mount, or enclose the target (e.g., tv stand, wall mount, shelf).
- "StrongCooccurs": nearby objects frequently co-existing with the target (e.g., sofa near tv, pillow near bed).
- "Gateways": entrances or passages guiding movement toward spaces likely containing the target.
  • Always include "doorway", "hallway", and "corridor".
  • Also add room-specific variants when sensible (e.g., "bathroom doorway", "kitchen doorway", "bedroom doorway").
- "Lookalikes": visually similar objects that may cause false detections (e.g., mirror for tv, trash bin for plant).

STYLE & CONSTRAINTS
1) Output lowercase noun phrases only; each item ≤ 3 words; no numbers; no hyphens/slashes; no duplicates.
2) Do not include the target itself or its direct synonyms in Supports/StrongCooccurs/Gateways. Synonyms or near-synonyms may appear in Lookalikes if they are genuinely confusable (e.g., "monitor" for "tv").
3) Prefer common, detector-friendly household terms (e.g., "tv stand", "coffee table", "nightstand", "sink").
4) No rooms in StrongCooccurs (put room access in Gateways). No explanations or extra keys.

LENGTH LIMITS
- "Supports": up to 10 items
- "StrongCooccurs": up to 10 items
- "Gateways": at least 4 and up to 10 items (must include "doorway","hallway","corridor")
- "Lookalikes": up to 10 items

OUTPUT FORMAT (critical)
- Return exactly one single-line JSON object with keys in this order:
  "Supports", "StrongCooccurs", "Gateways", "Lookalikes"
- Do not include "Answer:" wrappers, "Reason", prose, or backticks.
- No trailing commas. No additional text before/after the JSON.

Example style (values are illustrative only; adapt to the given target):
{"Supports":["tv stand","wall mount"],"StrongCooccurs":["sofa","coffee table","media console"],"Gateways":["doorway","hallway","corridor","living room doorway"],"Lookalikes":["monitor","picture frame","mirror"]}
"""
