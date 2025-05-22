import google.generativeai as genai #---->
genai.configure(api_key="AIzaSyBwrxR_jXcVCczEKom-nYUCPqvx9EReHuo")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Who is Modi?")
print(response.text)