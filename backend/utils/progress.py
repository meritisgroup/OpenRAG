import streamlit as st


class ProgressBar:
    def __init__(self, iterable=None, total=None, desc=None):
        self.iterable = iterable
        self.total = total or (len(iterable) if iterable is not None else None)
        self.desc = desc
        self.placeholder = st.empty()
        self.progress_bar = self.placeholder.progress(0, text=self.desc)
        self.progress_value = 0


    def set_description(self, desc):
        self.desc = desc
        if self.progress_bar:
            current_value = int(self.progress_value)
            self.progress_bar.progress(current_value, 
                                       text=self.desc)


    def update(self, index, text=None):
        try:
            if self.progress_bar:
                self.progress_value = int((index + 1) / self.total * 100)
                self.progress_bar.progress(
                    self.progress_value, text or self.desc
                )
                if self.progress_value >= 100:
                    self.clear()
            else:
                self.progress_bar = st.progress(
                    int((index + 1) / self.total * 100), text
                )
        except Exception as e:
            print(f"Error while uprading the progress bar : {e}")

    def message(self, text=None):
        st.write(text)

    def success(self, value):
        self.placeholder.empty()
        st.success(value)

    def error(self, value):
        self.placeholder.empty()
        st.error(value)

    def info(self, value):
        self.placeholder.info(value)

    def clear(self):
        self.progress_bar.empty()
        self.placeholder.empty()
