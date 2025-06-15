import requests
TOKEN = "8070914257:AAG5nSj9xPBHCQ4J-J3LATJrIFOAFDTUiVs"
ID = "-4764292106"
#chat id를 찾는 코드 
# print(requests.get(f"https://api.telegram.org/bot{token}/getUpdates").text)

def sendMessage(msg):
    requests.get((f"https://api.telegram.org/bot{TOKEN}/sendMessage"),params={'chat_id': f"{ID}", 'text': f"{msg}"})

if __name__ == "__main__":
    sendMessage("test")