from model import Model
from view import View
from controller import Controller
from config import Config

class App:
    def __init__(self):
        Config.load()
        self.model = Model()
        self.view = View()
        self.controller = Controller(self.model, self.view)
        self.view.set_controller(self.controller)

    def run(self):
        self.view.mainloop()

if __name__ == "__main__":
    app = App()
    app.run()