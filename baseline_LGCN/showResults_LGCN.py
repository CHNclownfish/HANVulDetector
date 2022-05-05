class showRes:
    def __init__(self,losses,scores):
        self.losses = losses
        self.scores = scores

    def showLoss(self):
        for loss in self.losses:
            print(loss)

    def showScores(self):
        for score in self.scores:
            print(score)