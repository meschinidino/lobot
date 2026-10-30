# Reader sample text

`reader_sample.txt` feeds the fires_on cache (R1). It holds 75,000
words of public-domain English. The Llama 3.2 3B tokenizer reads it
as 104,903 tokens. That clears the 50,000 token minimum.

## Sources

All five texts come from Project Gutenberg. Each is public domain
in the US.

| Ebook | Title | Author | Words used |
|---|---|---|---|
| 1342 | Pride and Prejudice | Jane Austen | 20,000 |
| 1661 | The Adventures of Sherlock Holmes | Arthur Conan Doyle | 20,000 |
| 2701 | Moby Dick; Or, The Whale | Herman Melville | 15,000 |
| 132 | The Art of War (Giles translation) | Sun Tzu | 10,000 |
| 1228 | On the Origin of Species, 1st edition | Charles Darwin | 10,000 |

URLs follow the pattern `https://www.gutenberg.org/files/<id>/<id>-0.txt`.
The mix covers fiction, dialogue, narrative, instructions, and
scientific prose.

## Assembly

1. Download the five `.txt` files from Project Gutenberg.
2. Keep the text between the `START OF THE PROJECT GUTENBERG`
   and `END OF THE PROJECT GUTENBERG` markers.
3. Take the first N words of each body (see the table).
4. Join the five parts with blank lines.

Step 3 cuts at a word boundary and keeps the original line breaks.
