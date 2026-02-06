pragma solidity ^0.5.0;

contract EthRoulette {
    uint256 private secretNumber;
    uint256 public lastPlayed;
    uint256 public betPrice = 0.1 ether;
    address payable public ownerAddr;
    
    struct Game {
        address player;
        uint256 number;
    }
    
    Game[] public gamesPlayed;
    
    constructor() public {
        ownerAddr = msg.sender;
        shuffle();
    }
    
    function shuffle() internal {
        secretNumber = (uint8(uint256(keccak256(abi.encodePacked(now, blockhash(block.number - 1))))) % 20) + 1;
    }
    
    function play(uint256 number) public payable {
        require(msg.value >= betPrice && number <= 20);
        Game memory game;
        game.player = msg.sender;
        game.number = number;
        gamesPlayed.push(game);
        if (number == secretNumber) {
            msg.sender.transfer(address(this).balance);
        }
        shuffle();
        lastPlayed = now;
    }
    
    function kill() public {
        if (msg.sender == ownerAddr && now > lastPlayed + 1 days) {
            selfdestruct(ownerAddr);
        }
    }
    
    function() external payable {}
}