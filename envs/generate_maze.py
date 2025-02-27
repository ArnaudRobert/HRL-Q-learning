import numpy as np
import matplotlib.pyplot as plt


class MazeGenerator():

    def __init__(self, maze_height, maze_width, room_height, room_width,
                 patterns=["", "U", "=", "||"]):
        """
        The environment is an array that obey the following structure.
        A 1 denotes a cell the agent can reach (from another adjacent cell).
        A 0 denotes a wall, an obstacle that prevent the agent to access this
        specific cell. -1 denotes the doors this are special cells that connect
        two rooms.

        To have access to all the information necessary to build the
        environment, this class construct three representations of the maze.
        The maze itself, the room_ids (an array that contain the room id of
        each rooms) and the states (an array that give to each different cell a
        unique id)
        """

        #check validity of params
        assert room_height % 2 == 1
        assert room_width % 2 == 1

        self.patterns = patterns
        self.mh = maze_height
        self.mw = maze_width
        self.rh = room_height
        self.rw = room_width
        self.state_offset = -1
        # placeholder
        self.maze = None
        self.room_ids = None
        self.states = None
        self.cell_states = {}
        self.obstacles = {}
        self._build_maze()

    def get_room_ids(self):
        return self.room_ids

    def get_state_ids(self):
        return self.state_ids

    def _get_base_room(self, close=""):
        """
        function to create a base room with a wall around
        and the four doors on each wall

        inputs:
            close  : specify the doors that are closed
        """
        height = self.rh
        width = self.rw
        # set the room id in each cell
        room = np.ones((height, width), dtype=np.int64)
        # set the wall around the room
        for i in np.arange(height):
            for j in np.arange(width):
                if i == 0 or i == height - 1:
                    room[i, j] = 0
                if j == 0 or j == width - 1:
                    room[i, j] = 0

        # set the door
        if 'N' not in close:
            room[0, width//2] = -1           # North door
        if 'W' not in close:
            room[height//2, 0] = -1          # West door
        if 'E' not in close:
            room[height//2, width - 1] = -1  # East door
        if 'S' not in close:
            room[height - 1, width//2] = -1  # South door

        return room

    def _build_maze(self):
        """
        given a collection of rooms build the maze

        inputs:
            height  : the height of the maze (in number of rooms)
            width   : the width of the maze (in number of rooms)
            rh      : height of a room
            rw      : width of a room
            profile : (int) the number of room profiles
        """
        height = self.mh * self.rh - (self.mh - 1)
        width = self.mw * self.rw - (self.mw - 1)
        self.maze = np.zeros((height, width))
        self.room_ids = np.zeros((height, width))
        self.states = np.zeros((height, width))

        k = 1  # add rooms to the maze
        for h in np.arange(self.mh):
            for w in np.arange(self.mw):
                close = ""
                if h % self.mh == 0:
                    close += "N"
                if w % self.mw == 0:
                    close += "W"
                if h % self.mh == (self.mh - 1):
                    close += "S"
                if w % self.mw == (self.mw - 1):
                    close += "E"

                room = self._get_base_room(close=close)
                p = np.random.randint(0, len(self.patterns))
                profile = self.patterns[p]
                if profile not in self.obstacles.keys():
                    self.cell_states[profile], self.obstacles[profile] = self._build_rooms_templates(profile)

                room[1:self.rh-1, 1:self.rw-1] = self.obstacles[profile]

                x1 = h * self.rh - h
                x2 = (h+1)*self.rh - h
                y1 = w * self.rw - w
                y2 = (w+1) * self.rw - w
                self.maze[x1:x2, y1:y2] = room
                self.room_ids[x1:x2, y1:y2] = room * k
                self.room_ids[self.room_ids < 0] = -1

                room[1:self.rh-1, 1:self.rw-1] = self.cell_states[profile] * self.obstacles[profile]
                self.states[x1:x2, y1:y2] = room
                k += 1

    def _save(self):
        path = "./mazes/"
        name = self._get_name()
        np.save(path+name+"_maze", self.maze)
        np.save(path+name+"_rooms", self.room_ids)
        np.save(path+name+"_states", self.states)


    def _get_name(self):
        name = f"{self.mh}x{self.mw}_rooms_{self.rh}x{self.rw}_cells_{len(self.patterns)}_profiles"
        return name

    def _save_fig(self):
        path = "figures/"
        fig, ax = self._show(show=False)
        name = self._get_name()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(path+name)
        plt.close()

    def _get_room_profiles(self, profile=""):
        room = self._get_base_room()
        room[room == -1] = 0.5
        room[1:self.rh-1, 1:self.rw-1] = self.obstacles[profile]
        return room

    def _get_maze(self):
        viz = self.maze.copy()
        viz *= -1
        viz[viz == 1] = -1 # mark doors as empty cell
        return viz

    def _show(self, show=True):

        viz = self.states.copy()
        viz *= -1
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        ax.imshow(viz, cmap="Greys")
        plt.axis('off')
        if show:
            plt.show()
        plt.close()
        viz = self.maze.copy()
        ax.grid('on')
        viz *= -1
        viz[viz == 1] = -1 # mark doors as empty cell
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        ax.imshow(viz, cmap="Greys")
        #plt.axis('off')
        if show:
            plt.show()

        return fig, ax

    def _build_rooms_templates(self, pattern):
        """
        manage the center of a rooms
        """
        # dimension on the interior of the room
        rh = self.rh - 2
        rw = self.rw - 2

        if pattern == "":  # no obstacle
            r = np.ones((rh, rw))
        if pattern == "U":
            r = np.ones((rh, rw))
            # add base
            r[rh-2, 1:rw-1] = 0
            # add lateral
            r[1:rh-1, 1] = 0
            r[1:rh-1, rh-2] = 0
        if pattern == "=":
            r = np.ones((rh, rw))
            r[1, 0:rw-1] = 0
            r[rh-2, 1:rw] = 0
        if pattern == '||':
            r = np.ones((rh, rw))
            r[0:rh-1, 1] = 0
            r[1:rh, rw-2] = 0

        s = r.cumsum().reshape(rh, rw) + self.state_offset
        self.state_offset = np.max(s)
        return s, r

if __name__ == '__main__':
    """
    Generate a maze. -1 encode walls, -2 encode doors
    and x>=0 encodes the room number
    """
    seed = 1
    m = 7  # square rooms with 7x7 cells
    np.random.seed(seed)
    Ns = [2, 4, 8, 16, 32, 64]  # square maze with NxN rooms
    patterns = ["", "U", "=", "||"]
    for N in Ns:
        for i in range(1, 5):
            gen = MazeGenerator(N, N, m, m, patterns[:i])
            #gen._plot_room_profiles()
            #gen._show()
            gen._save_fig()
            gen._save()

