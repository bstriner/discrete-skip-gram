class Watchdog(object):
    def __init__(self, limit, iters, path):
        self.limit = limit
        self.iters = iters
        self.counter = 0
        self.best_loss = None
        self.path = path
        self.resets = 0

    def check(self, loss):
        flag = False
        if self.best_loss:
            d = (self.best_loss - loss)
            if d < self.limit:
                self.counter += 1
            else:
                self.counter = 0
            if self.counter > self.iters:
                str = "[{}]: Improvement ({}) below limit (NLL: {})".format(
                    self.resets,
                    self.best_loss - loss,
                    loss)
                with open(self.path, 'a') as f:
                    f.write("{}\n".format(str))
                print(str)
                self.counter = 0
                self.best_loss = None
                self.resets += 1
                flag = True
            elif loss < self.best_loss:
                self.best_loss = loss
        else:
            self.best_loss = loss
        return flag
