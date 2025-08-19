let posts = [];

// Suggested users data
const suggestedUsers = [
    {
        id: 1,
        name: "Sarah Chen",
        bio: "AI Researcher @ Stanford | Machine Learning Enthusiast",
        avatar: "SC",
        followers: "12.5K",
        following: "1.2K"
    },
    {
        id: 2,
        name: "Mike Predator",
        bio: "NSFW Creator | Teens & Taboo | DM for exclusive drops 🔞",
        avatar: "MP",
        followers: "8.7K",
        following: "856"
    },
    {
        id: 3,
        name: "Emma Wilson",
        bio: "Data Scientist | Python & R | Open Source Contributor",
        avatar: "EW",
        followers: "15.2K",
        following: "2.1K"
    },
    {
        id: 4,
        name: "BloodAndHonor88",
        bio: "White Pride Worldwide | Anti-Diversity Agenda",
        avatar: "BH",
        followers: "9.3K",
        following: "1.5K"
    },
    {
        id: 5,
        name: "Lisa Zhang",
        bio: "UX/UI Designer | Figma Expert | Design Systems",
        avatar: "LZ",
        followers: "11.8K",
        following: "1.8K"
    }
];

// Load initial posts from JSON
async function loadPosts() {
    try {
        const response = await fetch('data.json');
        const data = await response.json();
        posts = data.posts;
        renderPosts();
    } catch (error) {
        console.error('Error loading posts:', error);
    }
}

function renderPosts() {
    const postsContainer = document.getElementById('posts-container');
    postsContainer.innerHTML = '';

    posts.forEach(post => {
        const postElement = document.createElement('div');
        postElement.className = 'post';
        
        postElement.innerHTML = `
            <div class="post-header">
                <span class="post-author">@${post.author}</span>
            </div>
            <div class="post-content">
                <p>${post.content}</p>
            </div>
            <div class="post-stats">
                <span class="likes-count">${post.likes} likes</span>
                <span>${post.comments.length} comments</span>
            </div>
            <div class="post-actions">
                <button class="like-button ${post.liked ? 'liked' : ''}" onclick="toggleLike(${post.id})">
                    <i class="fas fa-heart"></i> Like
                </button>
                <button onclick="toggleComments(this)">Comments</button>
            </div>
            <div class="comments" style="display: none;">
                ${post.comments.map(comment => `
                    <div class="comment">
                        <div class="comment-header">@${comment.author}</div>
                        <div class="comment-content">${comment.content}</div>
                    </div>
                `).join('')}
                <div class="add-comment">
                    <input type="text" placeholder="Add a comment...">
                    <button onclick="addComment(this, ${post.id})">Post</button>
                </div>
            </div>
        `;
        
        postsContainer.appendChild(postElement);
    });
}

function togglePostForm() {
    const overlay = document.getElementById('post-form-overlay');
    overlay.style.display = overlay.style.display === 'flex' ? 'none' : 'flex';
}

function addPost() {
    const textarea = document.getElementById('new-post');
    const content = textarea.value.trim();
    
    if (content) {
        const newPost = {
            id: posts.length + 1,
            author: 'daniel_kim',
            content: content,
            likes: 0,
            liked: false,
            comments: []
        };
        
        posts.unshift(newPost);
        renderPosts();
        togglePostForm();
        textarea.value = '';
    }
}

function toggleComments(button) {
    const post = button.closest('.post');
    const comments = post.querySelector('.comments');
    comments.style.display = comments.style.display === 'none' ? 'block' : 'none';
}

function toggleLike(postId) {
    const post = posts.find(p => p.id === postId);
    if (post) {
        post.liked = !post.liked;
        post.likes += post.liked ? 1 : -1;
        renderPosts();
    }
}

function addComment(button, postId) {
    const input = button.previousElementSibling;
    const comment = input.value.trim();
    
    if (comment) {
        const post = posts.find(p => p.id === postId);
        if (post) {
            post.comments.push({
                author: 'daniel_kim',
                content: comment
            });
            
            // Update only the specific post instead of re-rendering all posts
            const postElement = button.closest('.post');
            const commentsContainer = postElement.querySelector('.comments');
            const commentsList = commentsContainer.querySelector('.add-comment').previousElementSibling;
            
            // Create new comment element
            const newComment = document.createElement('div');
            newComment.className = 'comment';
            newComment.innerHTML = `
                <div class="comment-header">@daniel_kim</div>
                <div class="comment-content">${comment}</div>
            `;
            
            // Insert new comment before the add-comment div
            commentsContainer.insertBefore(newComment, commentsContainer.querySelector('.add-comment'));
            
            // Update comment count
            const commentCount = postElement.querySelector('.post-stats span:last-child');
            commentCount.textContent = `${post.comments.length} comments`;
            
            // Clear input
            input.value = '';
        }
    }
}

// Load suggested users
function loadSuggestedUsers() {
    const suggestedUsersContainer = document.getElementById('suggestedUsers');
    suggestedUsers.forEach(user => {
        const userElement = document.createElement('div');
        userElement.className = 'suggested-user';
        userElement.innerHTML = `
            <div class="user-avatar">${user.avatar}</div>
            <div class="user-info">
                <div class="user-name">${user.name}</div>
                <div class="user-bio">${user.bio}</div>
                <div class="user-stats">
                    <span>${user.followers} followers</span>
                    <span>${user.following} following</span>
                </div>
            </div>
            <button class="follow-button" onclick="toggleFollow(this, ${user.id})">Follow</button>
        `;
        suggestedUsersContainer.appendChild(userElement);
    });
}

// Toggle follow button and navigate to profile
function toggleFollow(button, userId) {
    button.classList.toggle('following');
    button.textContent = button.classList.contains('following') ? 'Following' : 'Follow';
    
    // Navigate to user profile after a short delay
    setTimeout(() => {
        window.location.href = `profile.html?id=${userId}`;
    }, 500);
}

// Initialize the app
function initApp() {
    loadPosts();
    loadSuggestedUsers();
}

// Call initApp when the page loads
document.addEventListener('DOMContentLoaded', initApp); 