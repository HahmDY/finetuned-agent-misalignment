from bs4 import BeautifulSoup


class BubbleEvaluator:
	def __init__(self):
		pass

	def get_comment_count(self, html, post_index):
		soup = BeautifulSoup(html, 'html.parser')

		comment_icons = soup.find_all('span', attrs={'data-text': '💭'})

		if post_index - 1 < 0 or post_index - 1 >= len(comment_icons):
			return None

		comment_icon = comment_icons[post_index - 1]
		comment_count_span = comment_icon.find_next_sibling('span')

		if comment_count_span is None:
			return None

		try:
			return int(comment_count_span.get_text(strip=True))
		except ValueError:
			return comment_count_span.get_text(strip=True)