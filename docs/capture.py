"""Capture README screenshots of the running app (python launch.py first).

    python docs/capture.py [--url http://127.0.0.1:8000] [--skip-chat]
"""
from __future__ import annotations

import argparse
import os

from playwright.sync_api import sync_playwright

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


def shot(page, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    page.screenshot(path=path, timeout=180_000)
    print("saved", path)


def open_sample(page, url, title):
    page.goto(url)
    page.get_by_text(title, exact=True).click(timeout=180_000)
    page.wait_for_selector(".part-name", timeout=120_000)
    page.wait_for_timeout(3500)          # camera ease-in + fade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--skip-chat", action="store_true")
    a = ap.parse_args()
    with sync_playwright() as p:
        p.selectors.set_test_id_attribute("data-testid")
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        for scheme in ("light", "dark"):
            ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2, color_scheme=scheme)
            page = ctx.new_page()
            page.goto(a.url)
            page.wait_for_timeout(1800)
            shot(page, f"welcome-{scheme}.png")

            open_sample(page, a.url, "L Bracket")
            shot(page, f"features-{scheme}.png")

            page.locator("button.run").click(timeout=120_000)
            page.wait_for_selector(".results", timeout=300_000)
            page.wait_for_timeout(1500)
            shot(page, f"results-{scheme}.png")
            page.locator(".results .icon-btn").click(timeout=120_000)
            page.wait_for_timeout(900)
            page.mouse.move(10, 450)
            shot(page, f"compare-{scheme}.png")

            if scheme == "light" and not a.skip_chat:
                open_sample(page, a.url, "Printer Chassis")
                page.get_by_text("Assistant", exact=True).click()
                page.wait_for_timeout(600)
                page.locator(".chat-input input").fill(
                    "Remove the text and all fillets, keep the screw bosses and the 7 mm mounting holes")
                page.locator(".chat-input input").press("Enter")
                page.wait_for_selector(".msg.assistant .calls", timeout=400_000)
                page.wait_for_timeout(1200)
                shot(page, "assistant-light.png")
            ctx.close()
        browser.close()


if __name__ == "__main__":
    main()
