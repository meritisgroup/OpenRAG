from backend.utils.md_docling import DoclingConverter


class MarkdownOpener:
    def __init__(
        self, config_server: dict, overwrite=False, image_description: bool = True
    ) -> None:
        self.overwrite = overwrite
        self.image_description = image_description
        self.config_server = config_server

    def open_doc(self, path_file) -> str:
        converter = DoclingConverter(self.config_server)
        markdown_content = converter.convert(
            path_file, image_description=self.image_description
        )
        return markdown_content
