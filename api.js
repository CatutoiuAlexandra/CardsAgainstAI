function api(path, method = "GET", body = null) {

    const url = "http://localhost:8080/cards-against-humanity/" + path;

    const options = {
        method,
        headers: {
            "Content-Type": "application/json; charset=utf-8",
        },
        mode:"cors",
    };
    if(body !== null){
        options.body = JSON.stringify(body);
    }
    return fetch(url, options);
}

var pickedBlackCard='';

async function fetchBlackCard() {
        return new Promise((resolve, reject) => {
            const randomIndex = Math.floor(Math.random() * blackCards.length);
            pickedBlackCard = blackCards[randomIndex];
            resolve(blackCards[randomIndex]);
        });

}

async function fetchWhiteCards() {
    return new Promise((resolve, reject) => {
        const shuffled = [...whiteCards].sort(() => 0.5 - Math.random()); // Amestecă lista
        resolve(shuffled.slice(0, 10));
    })

}


async function submitAIAnswer(blackCardText, candidateWhiteCards) {
    try {
        const response = await fetch('http://localhost:5000/pick', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                black_card_text: blackCardText,
                white_cards: candidateWhiteCards
            })
        });
        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Error:", error);
        return null;
    }
}
