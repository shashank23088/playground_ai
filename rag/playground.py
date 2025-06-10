# langchain-community module(s)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import FasterWhisperParser
from langchain_community.document_loaders import YoutubeAudioLoader


# YT audio loading

url = 'https://www.youtube.com/watch?v=TTTJ7mMtvlQ'    # Hardik Pandya
save_dir = './audios'
yt_loader = GenericLoader(
    YoutubeAudioLoader([url],save_dir),
    FasterWhisperParser(device='cpu')
)
docs = yt_loader.load()

print(docs)