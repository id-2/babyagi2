import openai
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
import os
import re
import importlib
import random


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
}

USER_AGENTS = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36 OPR/23.0.1522.60"},  # Opera on Windows
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.55 Safari/537.36"},  # Chrome on Windows
    {"User-Agent": "Mozilla/5.0 (Linux; Android 11; SM-A205U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.55 Mobile Safari/537.36"},  # Chrome on Android
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:95.0) Gecko/20100101 Firefox/95.0"},  # Firefox on Windows
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"},  # Chrome on Windows
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.55 Safari/537.36"},  # Chrome on Mac
    {"User-Agent": "Mozilla/5.0 (Linux; Android 11; SM-A205U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Mobile Safari/537.36"},  # Chrome on Android
    {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1"},  # Safari on iOS
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15"},  # Safari on Mac
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.48"},  # Edge on Windows
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:95.0) Gecko/20100101 Firefox/95.0"},  # Firefox on Windows
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:95.0) Gecko/20100101 Firefox/95.0"},  # Firefox on Mac
    {"User-Agent": "Mozilla/5.0 (Linux; Android 11; SM-A205U) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/14.0 Chrome/90.0.4430.210 Mobile Safari/537.36"}  # Samsung Internet on Android
]

# OS configuration
OBJECTIVE = os.getenv("OBJECTIVE", "")
INITIAL_TASK = os.getenv("INITIAL_TASK", "")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
CONTEXT_LENGTH = int(os.getenv("SUMMARY_CONTEXT", 3000))
SCRAPE_LENGTH = int(os.getenv("SCRAPE_LENGTH", 5000))

SUMMARY_CTX_MAX = int(os.getenv("SUMMARY_CTX_MAX", 1024))
LLAMA_THREADS_NUM = int(os.getenv("LLAMA_THREADS_NUM", 8))
SUMMARY_TEMPERATURE = float(os.getenv("SUMMARY_TEMPERATURE", 0.7))

# Report extension
ENABLE_REPORT_EXTENSION = os.getenv("ENABLE_REPORT_EXTENSION", "false").lower() == "true"
INSTRUCTION = os.getenv("INSTRUCTION", "")
ACTION = os.getenv("ACTION", "")


# API function: Search with SERPAPI, Google CSE or with browser (with fallback strategy to browser mode)
def web_search_tool(query: str, task: str, num_extracts: int, mode: str):
    links = []
    search_results = []
    results = []

    # Google API search
    if mode == "google":
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": query,
            "num": num_extracts,
            "start": 1
        }
        search_results = requests.get(url, params=params, timeout=5)

        if search_results.status_code == 200:
            try:
                json_data = search_results.json()
                if "items" in json_data:
                    search_results = json_data["items"]
                else:
                    print("Error: No items found in the response.")
                    search_results = []
                    if SERPAPI_API_KEY:
                        mode = "serpapi"
                        print("Switching to SERPAPI mode...")
                    else:
                        mode = "browser"
                        print("Switching to browser mode...")

            except ValueError as e:
                print(f"Error while parsing JSON data: {e}")

        else:
            print("\033[90m\033[3m" + f"Error: {search_results.status_code}\033[0m")
            if search_results.status_code == 429:
                if SERPAPI_API_KEY:
                    mode = "serpapi"
                    print("Switching to SERPAPI mode...")
                else:
                    mode = "browser"
                    print("Switching to browser mode...")
    
        print("\033[90m\033[3m" + "Completed search. Now scraping results...\n\033[0m")
        print_to_file("Completed search. Now scraping results...\n", 'a')
        links = []
        for result in search_results:
            links.append(result['link'])
            print("\033[90m\033[3m" + f"Webpage URL: {result['link']}\033[0m")

    # SERPAPI search
    if mode == "serpapi":
        search_params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": num_extracts
        }
        search_results = GoogleSearch(search_params)
        search_results = search_results.get_dict()

        try:
            search_results = search_results["organic_results"]
        except:
            search_results = {}
            mode = "browser"
            print("Switching to browser mode...")

        search_results = simplify_search_results(search_results)
        print("\033[90m\033[3m" + "Completed search. Now scraping results...\033[0m")
        print_to_file("Completed search. Now scraping results...\n", 'a')
        links = []
        for result in search_results:
            links.append(result.get('link'))
            print("\033[90m\033[3m" + f"Webpage URL: {result.get('link')}\033[0m")
    
    # Browser search
    if mode == "browser":
        access_counter = 0
        while (access_counter < 3):
            url = f"https://duckduckgo.com/html/?q={query}"
            browser_header = random.choice(USER_AGENTS)
            search_results = requests.get(url, headers=browser_header, timeout=5)
            if search_results.status_code == 200:
                try:
                    soup = BeautifulSoup(search_results.text, 'html.parser')
                    links = []
                    i = int(0)
                    for result in soup.select("a.result__url"):
                        url = result["href"]
                        if url:
                            links.append(url)
                            print("\033[90m\033[3m" + f"Webpage URL: {url}\033[0m")
                            i+=1
                        if i >= num_extracts:
                            access_counter = 3
                            break
                    
                    print("\033[90m\033[3m" + f"Completed search with {str(browser_header)}. Now scraping results...\n\033[0m")
                    print_to_file(f"Completed search with {str(browser_header)}. Now scraping results...\n", 'a')
                
                except Exception as e:
                    print("\033[90m\033[3m" + f"Error while parsing HTML data: {e}\033[0m")
                    access_counter = 3
                    links = []
                    search_results = []
            else:
                print(f"Request status code not OK: {search_results.status_code} with {browser_header}")
                access_counter+=1
                search_results = []

    # Error handling
    if mode not in ["google", "serpapi", "browser"]:
        print(f'Error: Smart search mode "{mode}" is out-of-range.')

    i = int(0)
    results=""
    # Scrape the search results
    if search_results:
        for result in search_results and links[i]:
            print("\033[90m\033[3m" + f"Scraping '{links[i]}'...\033[0m")
            print_to_file(f"Scraping '{links[i]}'...\n", 'a')
            content = web_scrape_tool(links[i], task)
            print("\033[90m\033[3m" + str(content) + "\n\033[0m")
            print_to_file(str(content) + "\n", 'a')
            results += str(content) + ". "
            i+=1
            if i >= num_extracts:
                break
    else:
        print("\033[90m\033[3m" + "No search results found.\033[0m")

    return results


### Tool functions ##############################
def can_import(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


# Write output to file
def print_to_file(text: str, mode: chr):
    with open('task_list.txt', mode) as f:
        f.write(text) 


# Function not used (yet)
def text_completion_tool(prompt: str):
    # Setup Llama model
    SUMMARY_MODEL_PATH = os.getenv("SUMMARY_MODEL_PATH", "")
    if SUMMARY_MODEL_PATH:
        if can_import("llama_cpp"):
            from llama_cpp import Llama

            print(f"LLAMA : {SUMMARY_MODEL_PATH}" + "\n")
            assert os.path.exists(SUMMARY_MODEL_PATH), "\033[91m\033[1m" + f"Text completion Llama model can't be found." + "\033[0m\033[0m"

            print('Initialize Llama model for text completion...')
            llm = Llama(
                model_path=SUMMARY_MODEL_PATH,
                n_ctx=1024,
                n_threads=LLAMA_THREADS_NUM,
                n_batch=512,
                use_mlock=False,
            )

            response = llm(prompt=prompt,
                            stop=["###"],
                            echo=False,
                            temperature=0.5,
                            top_k=40,
                            top_p=0.95,
                            repeat_penalty=1.05,
                            max_tokens=400)
            
            return response['choices'][0]['text'].strip()
                
        # Otherwise setup GPT-3 model
        else:
            messages = [
                {"role": "user", "content": prompt}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.2,
                max_tokens=1500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message['content'].strip()


def simplify_search_results(search_results):
    simplified_results = []
    for result in search_results:
        simplified_result = {
            "position": result.get("position"),
            "title": result.get("title"),
            "link": result.get("link"),
            "snippet": result.get("snippet")
        }
        simplified_results.append(simplified_result)
    return simplified_results


def web_scrape_tool(url: str, task:str):
    content = fetch_url_content(url)
    if content is None:
        return None

    text = extract_text(content)
    print("\033[90m\033[3m" + f"Scrape completed with length: {len(text)}. Now extracting relevant info with scrape length: {SCRAPE_LENGTH} and summary length: {CONTEXT_LENGTH}\033[0m")
    print_to_file(f"Scrape completed with length: {len(text)}. Now extracting relevant info with scrape length: {SCRAPE_LENGTH} and summary length: {CONTEXT_LENGTH}\n", 'a')
    info = extract_relevant_info(OBJECTIVE, text[0:SCRAPE_LENGTH], task)
    links = extract_links(content)

    #result = f"{info} URLs: {', '.join(links)}"
    result = info
    
    return result


def fetch_url_content(url: str):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error while fetching the URL: {e}")
        return ""


def extract_links(content: str):
    soup = BeautifulSoup(content, "html.parser")
    links = [link.get('href') for link in soup.findAll('a', attrs={'href': re.compile("^https?://")})]
    return links


def extract_text(content: str):
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text(strip=True)
    return text


def extract_relevant_info(objective, large_string, task):
    chunk_size = int(CONTEXT_LENGTH)
    overlap = int(chunk_size*0.1)
    notes = ""
    
    # Setup Llama model
    SUMMARY_MODEL_PATH = os.getenv("SUMMARY_MODEL_PATH", "")
    if SUMMARY_MODEL_PATH:
        if can_import("llama_cpp"):
            from llama_cpp import Llama

            print(f"LLAMA : {SUMMARY_MODEL_PATH}" + "\n")
            assert os.path.exists(SUMMARY_MODEL_PATH), "\033[91m\033[1m" + f"Search summary Llama model can't be found." + "\033[0m\033[0m"

            print('Initialize Llama model for search summary...')
            llm = Llama(
                model_path=SUMMARY_MODEL_PATH,
                n_ctx=SUMMARY_CTX_MAX,
                n_threads=LLAMA_THREADS_NUM,
                n_batch=512,
                use_mlock=False,
            )

            for i in range(0, len(large_string), chunk_size - overlap):
                chunk = large_string[i:i + chunk_size]
                if ENABLE_REPORT_EXTENSION and INITIAL_TASK not in task:
                    messages = f'Objective: {objective} Task: {task}\n'
                    messages += f'Consider the following action which shall be executed based on the objective, ignore for creation of a task list: {ACTION}\n'
                    messages += f'Consider the following instructions for execution of the action, ignore for creation of a task list: {INSTRUCTION}\n\n'
                else:
                    messages = f'Objective: {objective} Task: {task}\n\n'
                messages += f'Analyze the following text and extract information relevant to the objective and task, and only relevant information to the objective and task. Consider incomprehensible information as not relevant. If there is no relevant information do not say that there is no relevant information related to our objective. ### Then, update or start our notes provided here (keep blank if currently blank): {notes}. ### Text to analyze (ignore text if it is a string of keywords and not verbalized in sentences): {chunk}. ### Updated Notes: '
                response = llm(prompt=messages[0:SUMMARY_CTX_MAX],
                            stop=["###"],
                            echo=False,
                            temperature=SUMMARY_TEMPERATURE,
                            top_k=40,
                            top_p=0.95,
                            repeat_penalty=1.05,
                            max_tokens=400)
                
                notes += response['choices'][0]['text'].strip()+". "
                print(f"Search summary result for chunk '{i}':\n" + response['choices'][0]['text'].strip()+". ")
    
    # Otherwise setup GPT-3 model
    else:
        for i in range(0, len(large_string), chunk_size - overlap):
            chunk = large_string[i:i + chunk_size]
            messages = [
                {"role": "system", "content": f"Objective: {objective}\nCurrent Task:{task}\n"},
                {"role": "user", "content": f"Analyze the following text and extract information relevant to our objective and current task, and only information relevant to our objective and current task. Consider incomprehensible information as not relevant. If there is no relevant information do not say that there is no relevant information related to our objective. ### Then, update or start our notes provided here (keep blank if currently blank): {notes}. ### Text to analyze: {chunk}. ### Updated Notes: "}
            ]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=800,
                n=1,
                stop="###",
                temperature=0.7,
            )
            notes += response.choices[0].message['content'].strip()+". "
            #print(f"Search summary result for chunk '{i}':\n" + response['choices'][0]['text'].strip()+". ")

    return notes

