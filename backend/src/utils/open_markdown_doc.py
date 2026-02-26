from .md_conversion import VlmConverter

class MarkdownOpener:

    def __init__(self, agent, config_server: dict, overwrite=False, image_description: bool=True) -> None:
        self.overwrite = overwrite
        self.image_description = image_description
        self.config_server = config_server
        self.agent = agent
        self.converter = VlmConverter(self.config_server, agent=self.agent)

    def open_doc(self, path_file) -> str:
        markdown_content = self.converter.convert(path_file, max_workers=self.config_server['max_workers'], image_description=self.image_description)
        return markdown_content