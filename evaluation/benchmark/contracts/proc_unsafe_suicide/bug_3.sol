pragma solidity ^0.5.0;

contract ProofHashes {
    address payable public owner;
    
    constructor() public {
        owner = msg.sender;
    }
    
    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }
    
    function bug_intou27() public {
        uint8 vundflw = 0;
        vundflw = vundflw - 10;
    }
    
    event HashFormatSet(uint8 hashFunction, uint8 digestSize);
    
    function bug_intou31() public {
        uint8 vundflw = 0;
        vundflw = vundflw - 10;
    }
    
    event HashSubmitted(bytes32 hash);
    
    function _setMultiHashFormat(uint8 hashFunction, uint8 digestSize) internal {
        emit HashFormatSet(hashFunction, digestSize);
    }
    
    function bug_intou20(uint8 p_intou20) public {
        uint8 vundflw1 = 0;
        vundflw1 = vundflw1 + p_intou20;
    }
    
    function _submitHash(bytes32 hash) internal {
        emit HashSubmitted(hash);
    }
    
    function bug_intou32(uint8 p_intou32) public {
        uint8 vundflw1 = 0;
        vundflw1 = vundflw1 + p_intou32;
    }
    
    function kill() public onlyOwner {
        selfdestruct(owner);
    }
}