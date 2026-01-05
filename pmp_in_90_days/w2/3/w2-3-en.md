# 📦 Scrum Artifacts - Week 2, Session 3

## 🎯 Overview

After learning about the **5 Values**, **3 Roles**, and **4 Events** in Scrum, we now cover the **3 Artifacts** (outputs/products) that form the tangible foundation of Scrum work.

### 🔄 Quick Review of Previous Concepts

**Iteration vs Incremental:**
- **Iteration:** Reproducing the entire product with better quality until reaching the desired quality
- **Incremental:** Adding a small usable feature each time

**Practical Example of Incremental:**
Login page:
1. **Increment 1:** Simple page - email + password + login button (largest because it built the database and core technology)
2. **Increment 2:** Add "Login with Facebook" button
3. **Increment 3:** Add "Login with Google"
4. **Increment 4:** Add additional options

### 💎 Value

**Definition of Value:**
- Something that provides results or benefits to the user
- Can be financial (revenue)
- Can be non-financial (user satisfaction, reputation)
- Can be a mix of both
- **Must align with the organization's direction**

**Examples of Value:**
- 🥤 **Soft drink companies:** Value = sales revenue
- 💻 **Microsoft (historically):** Vision = a computer in every home running Windows
- 🍎 **Apple (historically):** Vision = everyone owns a Mac at home

## 📚 The 3 Scrum Artifacts

### 1️⃣ Product Backlog 📋

#### 🎯 Definition and Description

**Product Backlog** is:
- An ordered list of everything needed for the product
- A sorted and prioritized list
- Priority based on Value
- Value could be: Risk, Finance, Customer Satisfaction

#### 🔄 Product Backlog Characteristics

| Characteristic | Description | Notes |
|----------------|-------------|-------|
| **Dynamic** | Changes continuously | New items added after each Sprint |
| **Emergent** | Constantly improving | Top items are clear, bottom items less clear |
| **Single Source** | The only source of work | Scrum Team only works from this list |
| **Visible** | Clear to the entire team | Transparency is essential |

#### 📊 Product Backlog Structure

```
┌─────────────────────────────────────┐
│  Items at the Top                   │
│  ✅ Very clear                       │
│  ✅ Small (1-2 days)                 │
│  ✅ Ready to work on                 │
│  ✅ Estimated                        │
├─────────────────────────────────────┤
│  Items in the Middle                │
│  ⚠️ Moderately clear                │
│  ⚠️ Medium size                     │
├─────────────────────────────────────┤
│  Items at the Bottom                │
│  ❓ Not clear                        │
│  ❓ Very large                       │
│  ❓ Need detailing and refinement   │
└─────────────────────────────────────┘
```

#### 👤 Product Backlog Ownership

**Product Owner** is solely responsible for:
- ✅ Ordering the list by priority
- ✅ Writing items clearly and understandably
- ✅ Ensuring the team understands the items
- ✅ Adding and removing items
- ✅ Delegating tasks to others (but remains accountable)

⚠️ **Important Note:** 
- Product Owner is one person, not a group
- Must respect their decisions
- Any changes to the backlog must go through them

#### 🔧 Product Backlog Refinement

**What is Refinement?**
- Breaking large items into smaller ones
- Clarifying details
- Adding acceptance criteria
- Sizing/estimation
- Making items "Ready"

**Refinement Session Characteristics:**
- 📅 **Timing:** Continuous throughout the Sprint (not one-time)
- ⏰ **Duration:** As needed (30 minutes - 2 hours depending on complexity)
- 👥 **Participants:** Entire Scrum Team
- 🎯 **Goal:** Make items clear and ready for work

**Refinement Example:**
```
Before Refinement:
❌ "Develop products page" (large and unclear)

After Refinement:
✅ "Display product list" (1 day)
✅ "Add price filtering" (1 day)
✅ "Add sorting by popularity" (half day)
✅ "Add product search" (2 days)
```

#### 📏 Sizing/Estimation

**Who Estimates?**
- **Developers** estimate the size of tasks
- Reason: They will do the work, so they're best positioned to estimate

**Estimation Units:**
- Days
- Hours
- Story Points (most common in Agile)

#### ✅ Definition of Ready (DoR)

**When is a task "Ready" for work?**

| Criterion | Description |
|-----------|-------------|
| **No Blockers** | No road blocks preventing the start |
| **Clear** | Requirements are clear and understood |
| **Has Acceptance Criteria** | Clear acceptance criteria defined |
| **Estimated** | Size/time is known |
| **Has Skills** | Team has the required expertise |
| **Has Information** | All information and documentation available |

**Example of "Not Ready" Task:**
❌ "Integrate PayPal payment platform"
- No API documentation
- No contact person at PayPal
- Could take 1-2 weeks to get a response
- ❌ **Not Ready** - cannot start

---

### 2️⃣ Sprint Backlog 📝

#### 🎯 Definition

**Sprint Backlog** consists of:
1. **Sprint Goal** - Why?
2. **Product Backlog items** - What?
3. **Work Plan** - How?

```
Sprint Backlog = Sprint Goal + Selected Items + Plan
```

#### 📋 Sprint Backlog Components

**1. Sprint Goal:**
- Created at the beginning of the Sprint during Sprint Planning
- Clarifies the **Value** we'll achieve
- Written by **Developers** based on their work
- **Negotiable** with Product Owner

**2. Selected Items:**
- Tasks selected from **top of Product Backlog**
- Based on **Product Owner's priority**
- Based on **Team capacity**

**3. Work Plan:**
- **How** will we work?
- **Who** will work on what?
- Breaking tasks into smaller tasks

#### 👥 Sprint Backlog Ownership

| Aspect | Responsible |
|--------|-------------|
| **Full Ownership** | Developers |
| **Daily Updates** | Developers in Daily Scrum |
| **Planning** | Developers |
| **Execution** | Developers |
| **Accountability** | Developers |

⭐ **Important Point:** 
- Product Owner defines **What**
- Developers define **How**

#### 🔄 Sprint Backlog Dynamics

**Sprint Backlog evolves during the Sprint:**

```
Day 1:
├─ TODO: 10 tasks
├─ In Progress: 2 tasks
└─ Done: 0 tasks

Day 5:
├─ TODO: 4 tasks
├─ In Progress: 3 tasks  
└─ Done: 6 tasks

Day 10 (end of Sprint):
├─ TODO: 0 tasks
├─ In Progress: 0 tasks
└─ Done: 13 tasks (discovered 3 additional tasks)
```

#### 📊 Sufficient Detail in Sprint Backlog

**The backlog must contain sufficient detail:**
- ✅ Allow tracking progress in Daily Scrum
- ✅ Show status of each task (TODO / In Progress / Done)
- ✅ Show impediments if any
- ✅ Identify who's responsible for each task

#### 🎯 Commitment: Sprint Goal

**Sprint Goal Characteristics:**
- 🎯 Created during Sprint Planning
- 📌 Added to Sprint Backlog
- 👨‍💻 Set by Developers based on their work
- 🔄 If work differs from expected, can be negotiated

**Handling Change:**

| Situation | Action |
|-----------|--------|
| **Work less than expected** | Negotiate with Product Owner to add tasks |
| **Work more than expected** | Negotiate to reduce tasks and focus on goal |
| **Change in approach** | Negotiate to adjust tasks |

⚠️ **Golden Rule:** It's negotiable, but the **Goal is sacred** - must be achieved!

---

### 3️⃣ Increment 📦

#### 🎯 Definition

**Increment** is:
- A concrete, usable addition to the product
- Everything completed that meets the **Definition of Done**
- Must be **usable** even if not released

#### ✅ Basic Increment Requirements

| Requirement | Description |
|-------------|-------------|
| **Done** | Meets Definition of Done (DoD) |
| **Usable** | Can be used by customer |
| **Tested** | Tested and bug-free |
| **Integrated** | Integrated with previous increments |
| **Reviewable** | Ready to show in Sprint Review |

#### 🔄 Increment Cumulativeness

**Increments are cumulative:**

```
Sprint 1: Increment 1 = Login page
                      ↓
Sprint 2: Increment 2 = Increment 1 + Facebook login
                      ↓
Sprint 3: Increment 3 = Increment 2 + Google login
                      ↓
Sprint 4: Increment 4 = Increment 3 + Products page
```

**Each increment:**
- ✅ Adds to all previous increments
- ✅ Is verified
- ✅ Integrates with previous work
- ✅ Is usable

#### 📅 Increment Timing

**When is an Increment created?**
- ⏰ **At minimum:** Once at the end of each Sprint
- 🎯 **Better:** Multiple increments can be created during one Sprint
- 📦 **Presentation:** Shown in Sprint Review
- 🚀 **Release:** Can be released before Sprint end (if ready)

**Example:**
```
One Sprint (2 weeks):
├─ Increment 1: Login page (Day 3) ✅
├─ Increment 2: Add Facebook login (Day 6) ✅
├─ Increment 3: Products page (Day 10) ✅
└─ Sprint Review: Show all three increments
```

#### 🎯 Commitment: Definition of Done (DoD)

**What is Definition of Done?**
- 📋 Quality checklist
- 🎯 Release readiness indicator
- 📐 Formal standard agreed upon by all

#### 📊 Definition of Done vs Definition of Ready

| Comparison | Definition of Ready (DoR) | Definition of Done (DoD) |
|------------|---------------------------|-------------------------|
| **Timing** | Before starting work | After completing work |
| **Usage** | For planning and starting | For release and delivery |
| **Responsible** | Product Owner + Team | Entire team |
| **Goal** | Ensure task is clear | Ensure work is complete |

#### ✅ DoD Examples

**Definition of Done:**

1. ✅ **Work has been fully reviewed**
   - Completely reviewed by a colleague
   - Code review complete

2. ✅ **Work has been tested and no errors were found**
   - Given to Software Tester
   - Fully tested
   - Zero bugs

3. ✅ **Documentation is complete**
   - Technical documentation ready
   - User manual written

4. ✅ **Meets quality standards**
   - Meets agreed quality standards
   - Complies with coding standards

5. ✅ **Integrated with main codebase**
   - Merged into Main Branch
   - No conflicts

**Definition of Ready:**

1. ✅ **No Blockers**
   - No road blocks
   - Previous blockers resolved

2. ✅ **Clear Requirements**
   - Requirements clear and understood
   - Acceptance Criteria defined

3. ✅ **Team has Info and Skills**
   - Information available
   - Team has required expertise
   - Documentation ready

#### 🔍 Practical DoD Example

**Scenario: Add "Login with Facebook" feature**

```
❌ Not Done:
├─ Code written but not reviewed
├─ Testing incomplete
└─ Not merged into Main Branch

✅ Done (meets DoD):
├─ ✅ Code written and reviewed
├─ ✅ Full testing (Unit + Integration Tests)
├─ ✅ Merged into Main Branch
├─ ✅ Documentation written
├─ ✅ Shown to Product Owner
└─ ✅ Ready for user release

🚀 Now can be considered an Increment
```

#### ⚠️ Common Mistakes

| Mistake | Correction |
|---------|-----------|
| ❌ "Almost done, 90% ready" | ✅ Either Done or Not Done, no in-between |
| ❌ Delaying testing until Sprint end | ✅ Testing is part of daily work |
| ❌ Releasing something that doesn't meet DoD | ✅ No release until DoD is met |
| ❌ Different DoD for each person | ✅ One DoD agreed upon by everyone |

---

## 🔄 Complete Scrum Cycle with the 3 Artifacts

```
┌──────────────────────────────────────────────────────────┐
│                   Product Backlog                         │
│  (Product Owner's responsibility)                         │
│  ┌─────────┐                                             │
│  │ Task 1  │  ← High priority, clear, Ready               │
│  ├─────────┤                                             │
│  │ Task 2  │  ← Medium priority                           │
│  ├─────────┤                                             │
│  │ Task 3  │  ← Low priority, not clear                   │
│  └─────────┘                                             │
└──────────────────────────────────────────────────────────┘
                         ↓
              Sprint Planning
                         ↓
┌──────────────────────────────────────────────────────────┐
│                   Sprint Backlog                          │
│  (Developers' ownership)                                  │
│  ┌────────────────────────────────────────┐              │
│  │  Sprint Goal: Build login page         │              │
│  ├────────────────────────────────────────┤              │
│  │  TODO    │  In Progress  │  Done      │              │
│  ├──────────┼───────────────┼────────────┤              │
│  │  Task 3  │    Task 1     │            │  ← Day 1     │
│  │  Task 4  │    Task 2     │            │              │
│  └──────────┴───────────────┴────────────┘              │
│                                                          │
│         ↓ Daily Scrum (every day)                        │
│                                                          │
│  ┌──────────┬───────────────┬────────────┐              │
│  │  Task 4  │    Task 3     │  Task 1    │  ← Day 5     │
│  │          │                │  Task 2    │              │
│  └──────────┴───────────────┴────────────┘              │
└──────────────────────────────────────────────────────────┘
                         ↓
              Sprint Work
                         ↓
┌──────────────────────────────────────────────────────────┐
│                      Increment                            │
│  (Meets Definition of Done)                               │
│  ┌────────────────────────────────────────┐              │
│  │  ✅ Complete login page                 │              │
│  │  ✅ Tested and bug-free                 │              │
│  │  ✅ Integrated with previous release    │              │
│  │  ✅ Ready for release                   │              │
│  └────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────┘
                         ↓
              Sprint Review
                         ↓
         Sprint Retrospective
                         ↓
              Update Product Backlog
                         ↓
              New Sprint begins...
```

---

## 🎓 Other Agile Methodologies (Quick Overview)

### 🏭 Lean

**Origin:**
- From Japanese car manufacturing (Toyota)
- Emerged during the Great Depression
- While many companies closed, Toyota succeeded

**Core Principles:**
1. **Minimize Waste**
2. **Continuous Improvement**
3. **Optimize Processes**
4. **Focus on Value**

#### 🚫 The 7 Types of Waste

| Type | Description | Example |
|------|-------------|---------|
| **1. Overproduction** | Producing more than needed | Idle inventory |
| **2. Waiting** | Idle time | Waiting for approval, resources |
| **3. Inventory** | Excess storage | Large warehouses, costs |
| **4. Transportation** | Unnecessary movement | Distant warehouses |
| **5. Motion** | Excess employee movement | Going between offices |
| **6. Over-processing** | Using inefficient methods | Hammer instead of screwdriver |
| **7. Defects** | Poor quality products | Rework, repairs |

**Goal:** Reduce waste to zero or minimum possible

---

### 🔄 Kaizen

**Meaning:**
- Japanese word meaning "change for the better"
- Philosophy of continuous gradual improvement

**Kaizen Cycle (PDCA):**

```
    ┌──── Plan ────┐
    │              │
    │   🎯 Identify│
    │   📋 Plan    │
    │              │
    └──────┬───────┘
           │
           ↓
    ┌──── Do ─────┐
    │              │
    │   🔧 Execute │
    │   📊 Test    │
    │              │
    └──────┬───────┘
           │
           ↓
    ┌─── Check ───┐
    │              │
    │   ✅ Review  │
    │   📈 Measure │
    │              │
    └──────┬───────┘
           │
           ↓
    ┌──── Act ────┐
    │              │
    │   🎯 Standard│
    │   📚 Document│
    │              │
    └──────┬───────┘
           │
           └───────► Return to Plan for next improvement
```

**Principles:**
- 🔄 Small continuous improvements
- 👥 Employee participation at all levels
- 💡 Encourage suggestions from everyone
- 📊 Measure and document improvements

---

### 📊 Kanban

#### 🎯 Definition

**Kanban** (Japanese = "visual card")
- Visual system for workflow management
- Shows task status clearly
- Helps identify bottlenecks

#### 📋 Kanban Board

**Basic Format:**

```
┌──────────────────────────────────────────────────────────┐
│                  Kanban Board                             │
├─────────────┬─────────────────┬──────────────────────────┤
│   TODO      │  In Progress    │        Done              │
├─────────────┼─────────────────┼──────────────────────────┤
│             │                 │                          │
│ ┌─────────┐ │  ┌─────────┐   │   ┌─────────┐            │
│ │ Task 4  │ │  │ Task 1  │   │   │ Task 5  │            │
│ └─────────┘ │  └─────────┘   │   └─────────┘            │
│             │                 │                          │
│ ┌─────────┐ │  ┌─────────┐   │   ┌─────────┐            │
│ │ Task 3  │ │  │ Task 2  │   │   │ Task 6  │            │
│ └─────────┘ │  └─────────┘   │   └─────────┘            │
│             │                 │                          │
│ ┌─────────┐ │                 │                          │
│ │ Task 7  │ │                 │                          │
│ └─────────┘ │                 │                          │
│             │                 │                          │
└─────────────┴─────────────────┴──────────────────────────┘
```

#### 🎯 Key Kanban Characteristics

**1. Visualize Workflow:**
- Everyone sees work status clearly
- Easy tracking and monitoring

**2. Limit Work In Progress (WIP):**

```
┌─────────────┬──────────────────────┬─────────────┐
│   TODO      │  In Progress (WIP 5) │    Done     │
├─────────────┼──────────────────────┼─────────────┤
│ 10 tasks    │    5 tasks (limit)   │  15 tasks   │
└─────────────┴──────────────────────┴─────────────┘
```

**WIP = Work In Progress**
- **WIP 5** means: No more than 5 tasks in "In Progress"
- **Reason:** If you have 5 developers, shouldn't have more than 5 tasks in progress
- **Benefit:** Prevents starting new tasks without finishing old ones

**3. Manage Flow:**
- Identify bottlenecks
- Improve efficiency
- Reduce wait time

**4. Explicit Policies:**
- Clear rules for moving tasks between columns
- Agreed-upon criteria

**5. Feedback Loops:**
- Continuous review
- Process improvement

**6. Continuous Improvement:**
- Apply Kaizen to Kanban itself

#### 🏊 Swim Lanes

**Horizontal Division:**

```
┌───────────────────────────────────────────────────────────────┐
│                    Kanban Board                               │
├────────────┬─────────────────┬─────────────────┬──────────────┤
│            │      TODO       │  In Progress    │     Done     │
├────────────┼─────────────────┼─────────────────┼──────────────┤
│ Backend    │ API Task        │ DB Task         │ Login Task   │
├────────────┼─────────────────┼─────────────────┼──────────────┤
│ Frontend   │ UI Task         │ Form Task       │ Design Task  │
├────────────┼─────────────────┼─────────────────┼──────────────┤
│ Testing    │ Test Plan       │ Test Task       │ Report Task  │
└────────────┴─────────────────┴─────────────────┴──────────────┘
```

**Benefits:**
- ✅ Better organization by work type
- ✅ Clear responsibilities
- ✅ Easier team tracking

#### 📱 Physical vs Digital Kanban

| Physical Kanban | Digital Kanban |
|-----------------|----------------|
| ✅ Best for co-located teams | ✅ Best for distributed teams |
| ✅ Very easy to use (sticky notes) | ✅ Accessible from anywhere |
| ✅ High-Tech = touch! | ⚠️ Sometimes needs training |
| ✅ Everyone sees it constantly | ✅ Automatic reports |
| ⚠️ Difficult for remote teams | ✅ Historical tracking |

**Recommendation:**
- 🏆 **Best:** Physical Kanban on wall for co-located teams
- 💻 **Alternative:** Digital tools like Jira, Trello for distributed teams

#### 💡 Real-World Example

**Yasser's Story:**
```
Situation: Team at a Startup
Problem: Used Jira (advanced tool)
Result: Excessive complexity, team distraction

Solution: Return to physical Kanban
├─ Wall in team office
├─ Colored sticky notes
├─ TODO / In Progress / Done
└─ Clear WIP Limit

Result: 
✅ Simplicity
✅ Clarity
✅ Better communication
✅ Focus on completion
```

---

## 🎯 Session Summary

### 📦 The 3 Artifacts

| Artifact | Responsible | Purpose | Commitment |
|----------|-------------|---------|------------|
| **Product Backlog** | Product Owner | All required work | Product Goal |
| **Sprint Backlog** | Developers | Current Sprint work | Sprint Goal |
| **Increment** | Scrum Team | Usable result | Definition of Done |

### ✅ Always Remember

1. **Product Backlog:**
   - ✅ Dynamic and emergent
   - ✅ Ordered by priority/value
   - ✅ Top items are clear and ready

2. **Sprint Backlog:**
   - ✅ Fully owned by Developers
   - ✅ Contains Sprint Goal
   - ✅ Evolves during Sprint

3. **Increment:**
   - ✅ Must meet DoD
   - ✅ Usable
   - ✅ Cumulative (adds to previous)

### 🔄 Relationship Between Artifacts

```
Product Backlog (Everything) 
    ↓ (select from)
Sprint Backlog (Part for Sprint)
    ↓ (complete from)
Increment (Done result)
    ↓ (add to)
Product Backlog (update)
    ↓ (new Sprint)
...
```

### 🌟 Other Agile Methodologies

- **Lean:** Minimize waste, continuous improvement
- **Kaizen:** Continuous gradual improvement (PDCA)
- **Kanban:** Visual display, WIP Limit, Swim Lanes

---

## 📚 Important Terms

| English | Arabic |
|---------|--------|
| Product Backlog | قائمة المنتج |
| Sprint Backlog | قائمة السبرنت |
| Increment | الزيادة |
| Definition of Done (DoD) | تعريف الإنجاز |
| Definition of Ready (DoR) | تعريف الجاهزية |
| Backlog Refinement | تحسين القائمة |
| Estimation / Sizing | التقدير |
| Sprint Goal | هدف السبرنت |
| Product Goal | هدف المنتج |
| Impediments / Blockers | عوائق |
| Work In Progress (WIP) | العمل قيد التنفيذ |
| Lean | لين |
| Kaizen | كايزن |
| Kanban | كانبان |
| Waste | الهدر |
| Swim Lanes | المسارات |

---

## 🎓 For Review and Memorization

**Acronym for 3 Artifacts:** **PBI** (Product, Sprint, Increment)

**Acronym for Product Backlog characteristics:** **DEEP**
- **D**etailed appropriately
- **E**stimated
- **E**mergent
- **P**rioritized

**Acronym for 7 Wastes:** **TIMWOOD**
- **T**ransportation
- **I**nventory
- **M**otion
- **W**aiting
- **O**verproduction
- **O**ver-processing
- **D**efects

---

**Next Session:** We'll continue with more details on applying Scrum in real projects! 🚀
