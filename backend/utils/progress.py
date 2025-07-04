import streamlit as st


class ProgressBar:
    def __init__(self, iterable=None, total=None, desc=None):
        self.iterable = iterable
        self.total = total or (len(iterable) if iterable is not None else None)
        self.desc = desc
        self.progress_bar = st.progress(0, text=self.desc)
        self.status_placeholder = st.empty()

    def update(self, index, text=None):
        try:
            if self.progress_bar:
                self.progress_bar.progress(
                    int((index + 1) / self.total * 100), text or self.desc
                )
            else:
                self.progress_bar = st.progress(
                    int((index + 1) / self.total * 100), text
                )
        except Exception as e:
            print(f"Error while uprading the progress bar : {e}")

    def message(self, text=None):
        st.write(text)

    def success(self, value):
        self.progress_bar.empty()
        st.success(value)

    def error(self, value):
        self.progress_bar.empty()
        st.error(value)

    def info(self, value):
        self.status_placeholder.info(value)

    def clear(self):
        self.progress_bar.empty()
        self.status_placeholder.empty()
