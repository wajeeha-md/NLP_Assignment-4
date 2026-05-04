import urllib.request
import urllib.parse
import urllib.error
import asyncio

def _fetch_weather_sync(city: str) -> str:
    # Use wttr.in with format=3 (short format: City: Weather Condition +Temp)
    # We use format=j1 for json if we want structured data, but format=3 is very LLM friendly.
    # Let's use format=3. It returns something like "Lahore: ☀️   +35°C"
    safe_city = urllib.parse.quote(city)
    url = f"https://wttr.in/{safe_city}?format=3"
    
    try:
        # 5-second timeout requirement
        req = urllib.request.Request(url, headers={'User-Agent': 'curl/7.81.0'})
        with urllib.request.urlopen(req, timeout=5.0) as response:
            result = response.read().decode('utf-8').strip()
            if not result or "Unknown location" in result:
                return f"Error: Could not find weather for '{city}'."
            return result
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            return "Error: Weather service timed out."
        return f"Error: Failed to fetch weather ({e.reason})."
    except Exception as e:
        return f"Error: {str(e)}"

async def get_weather(city: str) -> str:
    """
    Asynchronously fetches the current weather for a specified city.
    
    Args:
        city: The name of the city (e.g., 'Lahore', 'New York')
        
    Returns:
        A string containing the weather info or an error message.
    """
    loop = asyncio.get_running_loop()
    # Run network request in a thread to prevent blocking the async loop
    result = await loop.run_in_executor(None, _fetch_weather_sync, city)
    return result

if __name__ == "__main__":
    async def test():
        import sys
        sys.stdout.reconfigure(encoding='utf-8')
        print("Testing valid city: 'Lahore' ->", await get_weather("Lahore"))
        print("Testing valid city: 'Karachi' ->", await get_weather("Karachi"))
        print("Testing invalid city: 'FakeCity12345' ->", await get_weather("FakeCity12345"))

    asyncio.run(test())
