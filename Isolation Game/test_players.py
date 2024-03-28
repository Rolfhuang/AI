from random import randint
import random
from isolation import Board
class Player():
    def __init__(self, name="Player"):
        self.name = name

    def move(self, game, time_left):
        pass

    def get_name(self):
        return self.name


class RandomPlayer(Player):
    """Player that chooses a move randomly."""
    def __init__(self, name="RandomPlayer"):
        super().__init__(name)

    def move(self, game, time_left):
        if not game.get_player_moves(self):
            return None
        else:
            return random.choice(game.get_player_moves(self))

    def get_name(self):
        return self.name


class HumanPlayer(Player):
    """
    Player that chooses a move according to user's input. 
    (Useful if you play in the terminal)
    """
    def __init__(self, name="HumanPlayer"):
        super().__init__(name)

    def move(self, game, time_left):
        legal_moves = game.get_player_moves(self)
        choice = {}

        if not len(legal_moves):
            print("No more moves left.")
            return None, None

        counter = 1
        for move in legal_moves:
            choice.update({counter: move})
            print('\t'.join(['[%d] (%d,%d)' % (counter, move[0], move[1])]))
            counter += 1

        print("-------------------------")
        print(game.print_board(legal_moves))
        print("-------------------------")
        print(">< - impossible, o - valid move")
        print("-------------------------")

        valid_choice = False

        while not valid_choice:
            try:
                index = int(input('Select move index [1-' + str(len(legal_moves)) + ']:'))
                valid_choice = 1 <= index <= len(legal_moves)

                if not valid_choice:
                    print('Illegal move of queen! Try again.')
            except Exception:
                print('Invalid entry! Try again.')

        return choice[index]

    def get_name(self):
        return self.name

class OpenMoveEvalFn:
    def score(self, game, my_player=None):
        """Score the current game state
        Evaluation function that outputs a score equal to how many
        moves are open for AI player on the board minus how many moves
        are open for Opponent's player on the board.

        Note:
            If you think of better evaluation function, do it in CustomEvalFn below.

            Args
                game (Board): The board and game state.
                my_player (Player object): This specifies which player you are.

            Returns:
                float: The current state's score. MyMoves-OppMoves.

            """

        # TODO: finish this function!͏︆͏󠄃͏󠄌͏󠄍͏︅͏︀͏︋͏︋͏󠄄͏︁͏︀
        my_move=game.get_player_moves(my_player)
        opponent_move=game.get_opponent_moves(my_player)
        score=len(my_move)-len(opponent_move)
        return score

class MiniMaxPlayer:
    # TODO: finish this class!͏︆͏󠄃͏󠄌͏󠄍͏︅͏︀͏︋͏︋͏󠄄͏︁͏︀
    """Player that chooses a move using your evaluation function
    and a minimax algorithm with alpha-beta pruning.
    You must finish and test this player to make sure it properly
    uses minimax and alpha-beta to return a good move."""

    def __init__(self, search_depth=3, eval_fn=OpenMoveEvalFn()):
        """Initializes your player.
        
        if you find yourself with a superior eval function, update the default
        value of `eval_fn` to `CustomEvalFn()`
        
        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Evaluation function used by your agent
        """
        self.eval_fn = eval_fn
        self.search_depth = search_depth
    
    def move(self, game, time_left):
        """Called to determine one move by your agent

        Note:
            1. Do NOT change the name of this 'move' function. We are going to call
            this function directly.
            2. Call alphabeta instead of minimax once implemented.
        Args:
            game (Board): The board and game state.
            time_left (function): Used to determine time left before timeout

        Returns:
            tuple: (int,int): Your best move
        """
        # best_move, utility = minimax(self, game, time_left, depth=self.search_depth)
        # # best_move, utility = alphabeta(self,game, time_left, depth=self.search_depth, alpha=float("-inf"), beta=float("inf"))
        # return best_move
        best_move=None
        current_depth=self.search_depth
        # start_time=time.time()
        # try:
        #     while True:
        #         if time_left()+start_time<50+start_time:
        #             break
        #         best_move, utility = alphabeta(self,game, time_left, depth=self.search_depth, alpha=float("-inf"), beta=float("inf"))
        #         # best_move, utility = minimax(self, game, time_left, depth=self.search_depth)
        #         # current_depth+=1
        #         return best_move
        # except TimeoutError:
        #     return best_move
        for i in range(current_depth):
            if time_left()<100: break
            best_move, utility = alphabeta(self,game, time_left, i, alpha=float("-inf"), beta=float("inf"))
        return best_move
    def utility(self, game, my_turn):
        """You can handle special cases here (e.g. endgame)"""
        return self.eval_fn.score(game, self)
    
def alphabeta(player, game, time_left, depth, alpha=float("-inf"), beta=float("inf"), my_turn=True):
    """Implementation of the alphabeta algorithm.
    
    Args:
        player (CustomPlayer): This is the instantiation of CustomPlayer() 
            that represents your agent. It is used to call anything you need 
            from the CustomPlayer class (the utility() method, for example, 
            or any class variables that belong to CustomPlayer())
        game (Board): A board and game state.
        time_left (function): Used to determine time left before timeout
        depth: Used to track how deep you are in the search tree
        alpha (float): Alpha value for pruning
        beta (float): Beta value for pruning
        my_turn (bool): True if you are computing scores during your turn.
        
    Returns:
        (tuple, int): best_move, val
    """
    legal_moves=game.get_active_moves()
    if depth==0 or not legal_moves:
        return None, player.utility(game, player)
    best_move=None
    if my_turn:
        for move in legal_moves:
            forecast_game, is_over, _ = game.forecast_move(move)
            if is_over:
                return move, player.utility(game, forecast_game)
            _, score = alphabeta(player,forecast_game, time_left, depth - 1, alpha, beta, False)
            if score > alpha:
                alpha = score
                best_move = move
            if alpha>=beta:
                break
        return best_move, alpha
    else:
        for move in legal_moves:
            forecast_game, is_over, _ = game.forecast_move(move)
            if is_over:
                return move, player.utility(game, forecast_game)
            _, score = alphabeta(player,forecast_game, time_left, depth - 1, alpha, beta, True)
            if score < beta:
                beta = score
                best_move = move
            if alpha>=beta:
                break
        return best_move, beta