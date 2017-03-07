"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


CENTER_SQUARE = [(x, y) for x in [2, 3, 4] for y in [2, 3, 4]]

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def center_moves(game, player):
    """Reward player's available moves in the center versus those of the
    opponent
    """
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    own_center_moves = len([m for m in own_moves if m in CENTER_SQUARE])
    opp_center_moves = len([m for m in opp_moves if m in CENTER_SQUARE])
    return float(own_center_moves - opp_center_moves)

def center_with_blank_moves(game, player):
    """Reward player's available moves in the center versus those of the
    opponent scaled according to the remaining blank spaces
    """
    blank_moves = len(game.get_blank_spaces())
    return float(center_moves(game, player) / blank_moves)

def moves_with_centers_and_blanks(game, player):
    """Calculates the ratio of the player moves + center moves minus the
    opponent moves + center moves to the number of
    free squares + player's moves - the opponent moves
    """
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    blank_moves = len(game.get_blank_spaces())
    own_center_moves = len([m for m in own_moves if m in CENTER_SQUARE])
    opp_center_moves = len([m for m in opp_moves if m in CENTER_SQUARE])
    p_moves = len(own_moves)
    o_moves = len(opp_moves)
    return ((p_moves + own_center_moves) - (o_moves + opp_center_moves) - blank_moves) / (blank_moves + p_moves - o_moves)

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    return moves_with_centers_and_blanks(game, player)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.) This parameter should be ignored when iterative = True.

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True). When True, search_depth should
        be ignored and no limit to search depth.

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        if not legal_moves:
            return (-1, -1)
        center = (game.width / 2, game.height / 2)
        # If available choose center move as best
        move = center if center in legal_moves else self._choose_random_move(legal_moves)
        search_method = self.minimax if self.method == 'minimax' else self.alphabeta

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            if not self.iterative:
                return move


            depth = 1
            while True:
                _, move = search_method(game, depth)
                depth += 1

        except Timeout:
            pass

        # Return the best move based on the last completed search iteration
        return move

    def _choose_random_move(self, legal_moves):
        return legal_moves[random.randint(0, len(legal_moves) - 1)]

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # If the game is already decided, then return the score with no legal move
        if game.utility(self) != 0.0:
            return game.utility(self), (-1, -1)

        moves = game.get_legal_moves()

        optimize = max if maximizing_player else min

        # Base case - at the bottom of the tree: calculate score for each legal move
        if depth == 1:
            return optimize([
                (self.score(game.forecast_move(m), self), m) for m in moves
            ])

        # Apply minimax at the next level and flip the optimizer (maximizing_player)
        return optimize([
            (self.minimax(game.forecast_move(m), depth - 1, not maximizing_player), m)
            for m in moves
        ])

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()


        moves, best_move = game.get_legal_moves(), (-1, -1)
        if not moves or depth <= 0:
            return self.score(game, self), best_move

        for move in moves:
            next_state = game.forecast_move(move)
            score, _ = self.alphabeta(next_state, depth - 1, alpha, beta, not maximizing_player)

            if maximizing_player:
                # New best score for the max player was found
                if score > alpha:
                    alpha = score
                    best_move = move
            else:
                # New best score for the min player was found
                if score < beta:
                    beta = score
                    best_move = move

            # We can prune the search tree if alpha > = beta
            if alpha >= beta:
                break
        score = alpha if maximizing_player else beta
        return score, best_move
