import os
import re
import time
import wikipedia
from tqdm import tqdm

wikipedia.set_lang("en")

TITLES_FILE = "data/articles_titles.txt"
OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_filename(title):
    return re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_").lower() + ".txt"


def clean_article(text):
    for marker in ["== References ==", "== See also ==", "== Notes ==",
                   "== External links ==", "== Further reading =="]:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_articles():
    with open(TITLES_FILE, "r", encoding="utf-8") as f:
        titles = [line.strip() for line in f if line.strip()]

    success, failed = 0, []

    for title in tqdm(titles, desc="Fetching articles"):
        try:
            page = wikipedia.page(title, auto_suggest=False)
            text = clean_article(page.content)
            fname = safe_filename(title)
            with open(os.path.join(OUTPUT_DIR, fname), "w", encoding="utf-8") as f:
                f.write(f"Title: {page.title}\n\n{text}")
            success += 1

        except wikipedia.DisambiguationError as e:
            options = [o for o in e.options if not o.lower().startswith("list")]
            if options:
                try:
                    page = wikipedia.page(options[0], auto_suggest=False)
                    text = clean_article(page.content)
                    fname = safe_filename(title)
                    with open(os.path.join(OUTPUT_DIR, fname), "w", encoding="utf-8") as f:
                        f.write(f"Title: {page.title}\n\n{text}")
                    success += 1
                except Exception as ex:
                    print(f"Failed: {title} -> {ex}")
                    failed.append(title)
            else:
                failed.append(title)

        except Exception as e:
            print(f"Error: {title}: {e}")
            failed.append(title)

        time.sleep(0.4)

    print(f"Done. Saved {success} articles. Failed: {failed or 'none'}")


if __name__ == "__main__":
    fetch_articles()
