// Get user ID from URL
const urlParams = new URLSearchParams(window.location.search);
const userId = urlParams.get('id');

// Find user data
const user = suggestedUsers.find(u => u.id === parseInt(userId));

// Update profile information
if (user) {
    document.getElementById('profileAvatar').textContent = user.avatar;
    document.getElementById('profileName').textContent = user.name;
    document.getElementById('profileBio').textContent = user.bio;
    
    const followersSpan = document.getElementById('profileFollowers');
    const followingSpan = document.getElementById('profileFollowing');
    
    followersSpan.setAttribute('data-count', user.followers);
    followersSpan.textContent = 'Followers';
    
    followingSpan.setAttribute('data-count', user.following);
    followingSpan.textContent = 'Following';
}

// Load user's posts
async function loadUserPosts() {
    try {
        const response = await fetch('data.json');
        const data = await response.json();
        const userPosts = data.posts.filter(post => post.author === user.name.toLowerCase().replace(' ', '_'));
        
        const postsContainer = document.getElementById('profilePosts');
        postsContainer.innerHTML = '';
        
        userPosts.forEach(post => {
            const postElement = document.createElement('div');
            postElement.className = 'post';
            postElement.innerHTML = `
                <div class="post-content">
                    <p>${post.content}</p>
                </div>
                <div class="post-stats">
                    <span class="likes-count">${post.likes} likes</span>
                    <span>${post.comments.length} comments</span>
                </div>
            `;
            postsContainer.appendChild(postElement);
        });
    } catch (error) {
        console.error('Error loading posts:', error);
    }
}

// Load posts when page loads
document.addEventListener('DOMContentLoaded', loadUserPosts); 