import random
from collections import defaultdict

class Tile:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value

    def __repr__(self):
        return f"{self.suit} {self.value}"

class MahjongGame:
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.players = [[] for _ in range(num_players)]
        self.wall = self._create_wall()
        self.discards = [[] for _ in range(num_players)]
        self.current_player = 0

    def _create_wall(self):
        # Create and shuffle a standard set of 144 tiles
        suits = ["mans", "pins", "sticks", "honors"]
        tiles = [Tile(suit, value) for suit in suits for value in range(1, 10)] * 4 #includes flowers
        random.shuffle(tiles)
        return tiles

    def deal_tiles(self):
        # Deal 13 tiles to each player
        for _ in range(13):
            for player in self.players:
                player.append(self.wall.pop())

    def draw_tile(self, player):
        # Player draws a tile from the wall
        tile = self.wall.pop()
        self.players[player].append(tile)
        return tile

    def discard_tile(self, player, tile):
        # Player discards a tile
        self.players[player].remove(tile)
        self.discards[player].append(tile)

    def check_win(self, player):
        # Check if the player's hand is a winning hand
        # Implement winning logic here
        pass

    def play_turn(self):
        # Simulate a player's turn
        tile = self.draw_tile(self.current_player)
        print(f"Player {self.current_player} draws {tile}")
        # Implement discard logic here
        self.current_player = (self.current_player + 1) % self.num_players

def is_straight(tiles):
    """
    Check if the given tiles form a valid straight (chow).
    Tiles must be of the same suit and consecutive in value.
    """
    if len(tiles) != 3:
        return False
    suits = {tile.suit for tile in tiles}
    if len(suits) != 1:  # All tiles must be of the same suit
        return False
    if tiles[0] == 'honors':
        return False
    values = sorted([tile.value for tile in tiles])
    return values[0] + 1 == values[1] and values[1] + 1 == values[2]

def is_pon(tiles):
    """
    Check if the given tiles form a valid pung (3 identical tiles).
    """
    if len(tiles) != 3:
        return False
    return all(tile == tiles[0] for tile in tiles)

def is_pair(tiles):
    """
    Check if the given tiles form a valid pair (2 identical tiles).
    """
    if len(tiles) != 2:
        return False
    return tiles[0] == tiles[1]

def find_melds(hand):
    """
    Find all possible melds (straights or pungs) in the hand.
    """
    melds = []
    hand = sorted(hand, key=lambda x: (x.suit, x.value))  # Sort hand for easier processing

    # Check for pungs
    counts = defaultdict(int)
    for tile in hand:
        counts[(tile.suit, tile.value)] += 1
    for key, count in counts.items():
        if count >= 3:
            melds.append([tile for tile in hand if tile.suit == key[0] and tile.value == key[1]][:3])

    # Check for straights
    for i in range(len(hand) - 2):
        group = hand[i:i+3]
        if is_straight(group):
            melds.append(group)

    return melds

def is_winning_hand(hand):
    """
    Check if the hand is a winning hand (4 melds and 1 pair).
    """
    if len(hand) != 14:  # A complete hand has 14 tiles
        return False

    # Find all possible pairs
    counts = defaultdict(int)
    for tile in hand:
        counts[(tile.suit, tile.value)] += 1
    pairs = [key for key, count in counts.items() if count >= 2]

    # Try each pair and see if the remaining tiles can form 4 melds
    for pair_key in pairs:
        remaining_hand = hand.copy()
        # Remove the pair from the hand
        pair_tiles = [tile for tile in remaining_hand if tile.suit == pair_key[0] and tile.value == pair_key[1]][:2]
        for tile in pair_tiles:
            remaining_hand.remove(tile)

        # Find melds in the remaining hand
        melds = find_melds(remaining_hand)
        if len(melds) >= 4:
            return True

    return False
# Example usage
game = MahjongGame()
game.deal_tiles()
game.play_turn()