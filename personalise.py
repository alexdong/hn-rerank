PROMPT = """
Analyze the following text and extract key concepts. Return a JSON where each
concept is assigned a weight (0-1) based **solely on its rarity on Hacker
News**, with 1.0 = rarest and 0.01 = most common. Normalize linearly.  

**Example Input**:  
"I am a theoretical biologist, interested in disease ecology. My tools are R,
Clojure, compartmentalism disease modeling, and statistical GAM models, using
data layers like geophysical, climate, biodiversity, and land use. I also work
on tech applied to agriculture, conservation, forecasting, third-world
development, AI, and large language models."  

**Example Output**:  
{
    "theoretical biology": 0.95,  
    "compartmentalism disease modeling": 0.85,  
    "statistical GAM_models": 0.80,  
    "geophysical data layers": 0.70,  
    "biodiversity data layers": 0.65,  
    "tech applied to conservation": 0.55,  
    "third world development": 0.35,  
    "AI applications": 0.25,  
    "large language models": 0.15
}  
"""

# write a function that takes a text and returns a dictionary of key concepts and weights, ai!
