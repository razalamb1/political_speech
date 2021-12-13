import requests
import time

## This file scrapes all posts and comments from three subreddits

# Define function to make single JSON request
def requestJSON(url):
    while True:
        try:
            r = requests.get(url)
            if r.status_code != 200:
                print("error code", r.status_code)
                time.sleep(5)
                continue
            else:
                break
        except Exception as e:
            print(e)
            time.sleep(5)
            continue
    return r.json()


# Define overall function to scrape
def scrape(subreddit, maxThings, printWait, requestSize):
    meta = requestJSON("https://api.pushshift.io/meta")
    limitPerMinute = meta["server_ratelimit_per_minute"]
    requestWait = 60 / limitPerMinute

    print("server_ratelimit_per_minute", limitPerMinute)

    things = ("submission", "comment")

    for thing in things:
        i = 0

        with open(
            "../02_text_files/" + subreddit + "_" + thing + ".txt",
            "w",
        ) as f:
            print("\n[starting", thing + "s]")

            if maxThings < 0:

                url = (
                    "https://api.pushshift.io/reddit/search/"
                    + thing
                    + "/?subreddit="
                    + subreddit
                    + "&metadata=true&size=0"
                )

                json = requestJSON(url)

                totalResults = json["metadata"]["total_results"]
                print("total " + thing + "s", "in", subreddit, ":", totalResults)
            else:
                totalResults = maxThings
                print("downloading most recent", maxThings)

            created_utc = ""

            startTime = time.time()
            timePrint = startTime
            while True:
                url = (
                    "http://api.pushshift.io/reddit/search/"
                    + thing
                    + "/?subreddit="
                    + subreddit
                    + "&size="
                    + str(requestSize)
                    + "&before="
                    + str(created_utc)
                )

                json = requestJSON(url)

                if len(json["data"]) == 0:
                    break

                doneHere = False
                for post in json["data"]:
                    created_utc = post["created_utc"]
                    f.write(str(post) + "\n")
                    i += 1
                    if i >= totalResults:
                        doneHere = True
                        break

                if doneHere:
                    break

                if time.time() - timePrint > printWait:
                    timePrint = time.time()
                    percent = i / totalResults * 100

                    timePassed = time.time() - startTime

                    print(
                        "{:.2f}".format(percent) + "%",
                        "|",
                        time.strftime("%H:%M:%S", time.gmtime(timePassed)),
                    )

                time.sleep(requestWait)


if __name__ == "__main__":
    scrape("democrats", -1, 2, 100)
    scrape("NeutralPolitics", -1, 2, 100)
    scrape("Republican", -1, 2, 100)
