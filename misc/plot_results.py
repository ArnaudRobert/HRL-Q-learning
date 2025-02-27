import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'DejaVu Sans',
        'serif': ['Computer Modern'],
        'size': 16}
matplotlib.rc('font', **font)

path = 'results'

def learning_curves():
    rooms = [2, 4, 8, 16, 32] #, 64]
    models = ['flat', 'hrl_stationary']
    labels = ['flat-RL', 'Stationary-HRL']
    colors = ['red', 'blue']
    linestyles = ['-', '-']
    smooth = 30
    with_obs = False

    fig, axes = plt.subplots(1, len(rooms), figsize=(21, 3))
    for i, r in enumerate(rooms):
        for im, m in enumerate(models):
            if with_obs:
                to_load = f"{path}/{m}_{r}x{r}_rooms_7x7_cells_with_obstacles_rooms.npy"
            else:
                to_load = f"{path}/{m}_{r}x{r}_rooms_7x7_cells_rooms.npy"
            print(f"reading:{to_load}")
            steps = np.load(f"{to_load}").squeeze()
            mean = np.mean(steps[:, :], axis=0)
            smooth_mean = np.convolve(mean, (1/smooth)*np.ones((smooth,)),mode='valid')
            std = np.std(steps[:, :], axis=0)
            truncated_std = std[smooth//2-1:-smooth//2]
            truncated_steps = np.arange(smooth//2-1, 3000-smooth//2)
            axes[i].plot(truncated_steps, smooth_mean, label=labels[im], color=colors[im],
                         linestyle=linestyles[im], linewidth=3, alpha=0.7)
            axes[i].fill_between(truncated_steps, smooth_mean-truncated_std,
                                 smooth_mean+truncated_std, alpha=0.3,
                                 color=colors[im])
            axes[i].set_ylim(0, np.max(mean))
            if i == 4:
                axes[i].legend(bbox_to_anchor=(1,1), loc="upper left")
            if i == 0:
                axes[i].set_ylabel('steps')
            axes[i].set_title(f'{r*r} rooms')
            axes[i].set_xlabel('episodes')

    plt.tight_layout()
    plt.savefig("figures/learning_curves.png")
    plt.show()



def state_space_impact(smooth=30):
    colors = ['black', 'red']
    linestyles = ['--', '-']

    # plot impact size of state space
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for i, r in enumerate([2, 4, 16]):
        name = f"hrl_large_low_level_state_space_qlearning_{r}_rooms.npy"
        steps = np.load(f"{path}/{name}")
        mean = np.mean(steps[:, 0, :], axis=0)
        axes[i].plot(np.convolve(mean, (1/smooth)*np.ones((smooth,)),
                     mode='valid'), linewidth=5, alpha=0.7, color=colors[1],
                     linestyle=linestyles[1],
                     label="HRL with large state space")

        name = f"{path}/flat_{r}_rooms.npy"
        steps = np.load(name)
        mean = np.mean(steps[:, 0, :], axis=0)
        axes[i].plot(np.convolve(mean, (1/smooth)*np.ones((smooth,)), mode='valid'),
                     color=colors[0], linestyle=linestyles[0],
                     linewidth=5, alpha=0.7, label="No-hierarchy")
        if i == 2:
            axes[i].legend(bbox_to_anchor=(1, 1), loc="upper left")

        if i == 0:
            axes[i].set_ylabel('steps')
        axes[i].set_title(f'{r} rooms')
        axes[i].set_xlabel('episodes')

    plt.tight_layout()
    plt.savefig("figures/large_state_space")
    plt.show()


def plot_low_level_policies():
    Ql = np.load("results/hrl_qlearning_9_rooms_low_level_Q.npy")
    vmin = np.min(Ql)
    vmax = np.max(Ql)
    titles = ['North exit', 'East exit',
              'South exit', 'West exit']
    room = np.zeros((7, 7))
    room[0, :] = -10
    room[:, 0] = -10
    room[6, :] = -10
    room[:, 6] = -10
    room[0, 3] = 10
    room[3, 0] = 10
    room[6, 3] = 10
    room[3, 6] = 10

    mask = room.copy()

    for i in range(4):
        fig, ax = plt.subplots(1, 1)

        # ax.imshow(np.max(Ql[i, :, :], axis=1).reshape(5, 5),
        #          vmin=vmin, vmax=vmax, interpolation='nearest')
        #room[1:6, 1:6] = np.max(Ql[i, :, :], axis=1).reshape(5,5)
        room[1:6, 1:6] = np.max(Ql[i, :, :], axis=1).reshape(5, 5)
        cmap = matplotlib.colormaps['viridis']
        cmap.set_under('k')
        cmap.set_over('white')
        ax.imshow(room, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8)
        #ax.imshow(mask, cmap=,
        #          vmin=0, vmax=1, alpha=1)

        ax.tick_params(left=False,
                       bottom=False,
                       labelleft=False,
                       labelbottom=False)
        #ax.set_title(titles[i])
        plt.tight_layout()
        plt.savefig(f"./figures/low_level_policy_{titles[i]}")
    plt.show()


def plot_high_level_policies():
    from rooms_for_hrl import Fourrooms
    from mazes.config import maze3x3
    env = Fourrooms(maze3x3)
    env.rooms[env.rooms >= 0] = 0
    env.rooms[env.rooms == -1] = 1
    env.rooms[env.rooms == -2] = 0
    fig, ax = plt.subplots(1, 1)
    Qh = np.load("results/hrl_qlearning_9_rooms_high_level_Q.npy")
    print(np.argmax(Qh, axis=1))
    ax.imshow(env.rooms, vmin=0, vmax=1, cmap=matplotlib.colormaps['Greys'])
    ax.arrow(3, 3, 0, 1, width=0.35, head_width=0.75, color='k')
    ax.arrow(9, 3, 0, 1, width=0.35, head_width=0.75, color='k')
    ax.arrow(15, 3, 0, 1, width=0.35, head_width=0.75, color='k')

    ax.arrow(3, 9, 1, 0, width=0.35, head_width=0.75, color='k')
    ax.arrow(9, 9, 1, 0, width=0.35, head_width=0.75, color='k')
    ax.arrow(15, 9, 0, 1, width=0.35, head_width=0.75, color='k')

    ax.arrow(3, 15, 1, 0, width=0.35, head_width=0.75, color='k')
    ax.arrow(9, 15, 1, 0, width=0.35, head_width=0.75, color='k')

    ax.tick_params(left=False,
                   bottom=False,
                   labelleft=False,
                   labelbottom=False)

    plt.tight_layout()
    plt.savefig("./figures/high_level_policy")
    plt.show()


def plot_transition_usage():
    fig, ax = plt.subplots(1,1)
    ts = np.load("./results/hrl_strict_9_rooms_used_transitions.npy")
    ax.plot(np.convolve(np.ones((30,)), ts, mode='valid')* 1/30,
            linewidth=5, alpha=0.8)
    ax.set_title("ratio of used transition for high level updates")
    ax.set_ylabel("ration of transition")
    plt.show()



if __name__ == '__main__':
    learning_curves()
    #state_space_impact()
    #plot_low_level_policies()
    #plot_high_level_policies()
    #plot_transition_usage()
