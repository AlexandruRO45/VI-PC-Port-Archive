import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window
    width: 1480
    height: 920
    minimumWidth: 1120
    minimumHeight: 700
    visible: true
    title: "VIctory — Grand Theft Auto VI"
    color: "#0b0d10"
    font.family: "Segoe UI"

    property int pageIndex: 0
    readonly property color paper: "#f2eee8"
    readonly property color muted: "#a09b94"
    readonly property color accent: "#d83b4f"
    readonly property color surface: "#15171b"
    readonly property color edge: "#2a2d32"

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        ColumnLayout {
            id: workspace
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 70
                color: "#0f1114"
                border.color: "#25282c"
                border.width: 1
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 24
                    anchors.rightMargin: 24
                    spacing: 14
                    RowLayout {
                        Layout.preferredWidth: 190
                        spacing: 10
                        Rectangle {
                            width: 36
                            height: 36
                            radius: 9
                            color: "#282026"
                            border.color: "#4b3039"
                            Text {
                                anchors.centerIn: parent
                                text: "VI"
                                color: "#f0d7c6"
                                font.family: "Georgia"
                                font.pixelSize: 16
                            }
                        }
                        ColumnLayout {
                            spacing: 1
                            Text {
                                text: "VICTORY"
                                color: window.paper
                                font.pixelSize: 12
                                font.weight: Font.Bold
                                font.letterSpacing: 1.5
                            }
                            Text {
                                text: "ROCKSTAR COLLECTION"
                                color: "#777b7e"
                                font.pixelSize: 8
                                font.weight: Font.DemiBold
                                font.letterSpacing: 0.7
                            }
                        }
                    }
                    Rectangle { width: 1; height: 30; color: "#303338" }
                    RowLayout {
                        spacing: 5
                        Repeater {
                            model: [
                                { label: "Library", page: 0 },
                                { label: "Activity", page: 1 },
                                { label: "Settings", page: 2 }
                            ]
                            delegate: Rectangle {
                                required property var modelData
                                Layout.preferredWidth: 94
                                Layout.preferredHeight: 40
                                radius: 7
                                color: window.pageIndex === modelData.page ? "#242025" : topNavMouse.containsMouse ? "#191b1f" : "transparent"
                                Text {
                                    anchors.centerIn: parent
                                    text: modelData.label
                                    color: window.pageIndex === modelData.page ? "#f0e6e0" : "#a0a2a4"
                                    font.pixelSize: 12
                                    font.weight: window.pageIndex === modelData.page ? Font.DemiBold : Font.Normal
                                }
                                Rectangle {
                                    visible: window.pageIndex === modelData.page
                                    width: 24
                                    height: 2
                                    radius: 1
                                    color: window.accent
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    anchors.bottom: parent.bottom
                                }
                                MouseArea {
                                    id: topNavMouse
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: window.pageIndex = modelData.page
                                }
                            }
                        }
                    }
                    Item { Layout.fillWidth: true }
                    Rectangle {
                        Layout.preferredWidth: launcher.running ? 112 : 86
                        Layout.preferredHeight: 30
                        radius: 15
                        color: "#191b1e"
                        border.color: "#34373b"
                        RowLayout {
                            anchors.centerIn: parent
                            spacing: 7
                            Rectangle {
                                width: 6
                                height: 6
                                radius: 3
                                color: launcher.running ? "#e2ac71" : "#7fb78a"
                            }
                            Text {
                                text: launcher.running ? "IN SESSION" : "LOCAL"
                                color: "#b9b8b4"
                                font.pixelSize: 9
                                font.weight: Font.Bold
                                font.letterSpacing: 1
                            }
                        }
                    }
                    Rectangle {
                        width: 34
                        height: 34
                        radius: 17
                        color: "#25272b"
                        border.color: "#45484c"
                        Text {
                            anchors.centerIn: parent
                            text: "V"
                            color: "#e4c4ae"
                            font.family: "Georgia"
                            font.pixelSize: 15
                        }
                    }
                }
            }

            StackLayout {
                id: pages
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: window.pageIndex

                Item {
                    Flickable {
                        anchors.fill: parent
                        anchors.leftMargin: 28
                        anchors.rightMargin: 28
                        anchors.topMargin: 24
                        anchors.bottomMargin: 22
                        contentWidth: width
                        contentHeight: libraryContent.implicitHeight
                        clip: true
                        boundsBehavior: Flickable.StopAtBounds

                        ColumnLayout {
                            id: libraryContent
                            width: parent.width
                            spacing: 17

                            RowLayout {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 48
                                ColumnLayout {
                                    spacing: 2
                                    Text {
                                        text: "YOUR COLLECTION"
                                        color: "#86898c"
                                        font.pixelSize: 9
                                        font.weight: Font.Bold
                                        font.letterSpacing: 1.8
                                    }
                                    Text {
                                        text: "Game library"
                                        color: window.paper
                                        font.family: "Georgia"
                                        font.pixelSize: 25
                                    }
                                }
                                Item { Layout.fillWidth: true }
                                Text {
                                    text: "01  TITLE"
                                    color: "#85888c"
                                    font.pixelSize: 10
                                    font.weight: Font.DemiBold
                                    font.letterSpacing: 1.1
                                }
                            }

                            Rectangle {
                                id: featureCard
                                Layout.fillWidth: true
                                Layout.preferredHeight: 374
                                radius: 14
                                color: "#17191d"
                                border.color: "#33363a"
                                clip: true

                                RowLayout {
                                    anchors.fill: parent
                                    spacing: 0
                                    Item {
                                        Layout.fillWidth: true
                                        Layout.fillHeight: true
                                        Layout.preferredWidth: featureCard.width * 0.59
                                        Layout.minimumWidth: 300
                                        clip: true
                                        Image {
                                            anchors.fill: parent
                                            source: launcher.wallpaperSource
                                            fillMode: Image.PreserveAspectCrop
                                            sourceSize.width: 3840
                                            sourceSize.height: 2160
                                            asynchronous: true
                                            smooth: true
                                        }
                                        Rectangle {
                                            anchors.fill: parent
                                            gradient: Gradient {
                                                orientation: Gradient.Horizontal
                                                GradientStop { position: 0.0; color: "#12090b10" }
                                                GradientStop { position: 0.66; color: "#3b090b10" }
                                                GradientStop { position: 1.0; color: "#b90a0c10" }
                                            }
                                        }
                                        Rectangle {
                                            anchors.left: parent.left
                                            anchors.top: parent.top
                                            anchors.leftMargin: 20
                                            anchors.topMargin: 20
                                            width: 144
                                            height: 29
                                            radius: 15
                                            color: "#bb101215"
                                            border.color: "#6e74777a"
                                            RowLayout {
                                                anchors.centerIn: parent
                                                spacing: 7
                                                Rectangle { width: 6; height: 6; radius: 3; color: window.accent }
                                                Text {
                                                    text: "ROCKSTAR GAMES"
                                                    color: "#e2ded9"
                                                    font.pixelSize: 9
                                                    font.weight: Font.Bold
                                                    font.letterSpacing: 0.9
                                                }
                                            }
                                        }
                                        ColumnLayout {
                                            anchors.left: parent.left
                                            anchors.bottom: parent.bottom
                                            anchors.leftMargin: 24
                                            anchors.bottomMargin: 23
                                            spacing: 5
                                            Text {
                                                text: "VICE CITY  ·  LEONIDA"
                                                color: "#f1c8ad"
                                                font.pixelSize: 10
                                                font.weight: Font.Bold
                                                font.letterSpacing: 1.6
                                            }
                                            Text {
                                                text: "A new chapter begins."
                                                color: "#f2eee9"
                                                font.family: "Georgia"
                                                font.pixelSize: 21
                                            }
                                        }
                                    }

                                    Rectangle {
                                        Layout.fillHeight: true
                                        Layout.fillWidth: true
                                        Layout.preferredWidth: featureCard.width * 0.41
                                        Layout.minimumWidth: 270
                                        color: "#17191d"
                                        ColumnLayout {
                                            anchors.fill: parent
                                            anchors.leftMargin: 24
                                            anchors.rightMargin: 24
                                            anchors.topMargin: 23
                                            anchors.bottomMargin: 22
                                            spacing: 0
                                            RowLayout {
                                                Layout.fillWidth: true
                                                Rectangle {
                                                    Layout.preferredWidth: launcher.gameExists ? 76 : 90
                                                    Layout.preferredHeight: 25
                                                    radius: 13
                                                    color: launcher.gameExists ? "#1c3025" : "#332521"
                                                    border.color: launcher.gameExists ? "#365846" : "#614137"
                                                    Text {
                                                        anchors.centerIn: parent
                                                        text: launcher.gameExists ? "READY" : "SETUP"
                                                        color: launcher.gameExists ? "#9dcca4" : "#e0ad8d"
                                                        font.pixelSize: 9
                                                        font.weight: Font.Bold
                                                        font.letterSpacing: 1
                                                    }
                                                }
                                                Item { Layout.fillWidth: true }
                                                Text {
                                                    text: "ROCKSTAR  /  VI"
                                                    color: "#85888b"
                                                    font.pixelSize: 9
                                                    font.weight: Font.DemiBold
                                                    font.letterSpacing: 0.9
                                                }
                                            }
                                            Text {
                                                Layout.fillWidth: true
                                                Layout.topMargin: 17
                                                text: "Grand Theft\nAuto VI"
                                                color: window.paper
                                                font.family: "Georgia"
                                                font.pixelSize: 34
                                                lineHeight: 0.92
                                            }
                                            Text {
                                                Layout.fillWidth: true
                                                Layout.topMargin: 9
                                                text: launcher.gameExists ? "Your game is ready. Start a session in " + launcher.consoleModel + "." : "Explore the state of Leonida with Jason and Lucia."
                                                color: "#a8a6a1"
                                                font.pixelSize: 12
                                                lineHeight: 1.25
                                                wrapMode: Text.WordWrap
                                            }
                                            Item { Layout.fillHeight: true }
                                            Text {
                                                Layout.fillWidth: true
                                                text: launcher.hasGame ? launcher.selectedPath : "Add the game executable from your existing install."
                                                color: "#86888a"
                                                font.pixelSize: 10
                                                elide: Text.ElideMiddle
                                            }
                                            Rectangle {
                                                Layout.fillWidth: true
                                                Layout.topMargin: 11
                                                Layout.preferredHeight: 51
                                                radius: 8
                                                color: playMouse.containsMouse ? "#e34b5c" : window.accent
                                                RowLayout {
                                                    anchors.fill: parent
                                                    anchors.leftMargin: 15
                                                    anchors.rightMargin: 15
                                                    spacing: 9
                                                    Text {
                                                        text: launcher.running ? "●" : "▶"
                                                        color: "white"
                                                        font.pixelSize: 11
                                                    }
                                                    Text {
                                                        Layout.fillWidth: true
                                                        horizontalAlignment: Text.AlignHCenter
                                                        text: launcher.running ? "SESSION IN PROGRESS" : launcher.gameExists ? "PLAY GRAND THEFT AUTO VI" : launcher.hasGame ? "LOCATE GAME FILES" : "ADD GAME FILES"
                                                        color: "white"
                                                        font.pixelSize: 11
                                                        font.weight: Font.Bold
                                                        font.letterSpacing: 0.8
                                                        elide: Text.ElideRight
                                                    }
                                                    Item { width: 12; height: 1 }
                                                }
                                                MouseArea {
                                                    id: playMouse
                                                    anchors.fill: parent
                                                    hoverEnabled: true
                                                    cursorShape: Qt.PointingHandCursor
                                                    enabled: !launcher.running
                                                    onClicked: launcher.gameExists ? launcher.launch() : launcher.browse()
                                                }
                                            }
                                            Rectangle {
                                                Layout.fillWidth: true
                                                Layout.topMargin: 8
                                                Layout.preferredHeight: 34
                                                radius: 7
                                                color: locateMouse.containsMouse ? "#22252a" : "transparent"
                                                Text {
                                                    anchors.centerIn: parent
                                                    text: launcher.hasGame ? "CHANGE GAME LOCATION" : "I'LL ADD FILES LATER"
                                                    color: "#aaa69f"
                                                    font.pixelSize: 9
                                                    font.weight: Font.DemiBold
                                                    font.letterSpacing: 0.8
                                                }
                                                MouseArea {
                                                    id: locateMouse
                                                    anchors.fill: parent
                                                    hoverEnabled: true
                                                    cursorShape: Qt.PointingHandCursor
                                                    onClicked: launcher.browse()
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 174
                                spacing: 12

                                Rectangle {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    radius: 11
                                    color: window.surface
                                    border.color: window.edge
                                    ColumnLayout {
                                        anchors.fill: parent
                                        anchors.margins: 15
                                        spacing: 9
                                        Text {
                                            text: "GAME INSTALLATION"
                                            color: "#888b8f"
                                            font.pixelSize: 9
                                            font.weight: Font.Bold
                                            font.letterSpacing: 1.15
                                        }
                                        Text {
                                            Layout.fillWidth: true
                                            text: launcher.hasGame ? launcher.gameTitle : "Game files not added"
                                            color: window.paper
                                            font.family: "Georgia"
                                            font.pixelSize: 16
                                            elide: Text.ElideRight
                                        }
                                        Text {
                                            Layout.fillWidth: true
                                            Layout.fillHeight: true
                                            text: launcher.hasGame ? launcher.selectedPath : "Select the GTA VI executable in your existing folder."
                                            color: window.muted
                                            font.pixelSize: 10
                                            elide: Text.ElideMiddle
                                            wrapMode: Text.Wrap
                                        }
                                        Rectangle {
                                            Layout.preferredWidth: 122
                                            Layout.preferredHeight: 29
                                            radius: 6
                                            color: installMouse.containsMouse ? "#282b30" : "#202328"
                                            border.color: "#373b40"
                                            Text {
                                                anchors.centerIn: parent
                                                text: launcher.hasGame ? "CHANGE LOCATION" : "SELECT FILES"
                                                color: "#d9d3cb"
                                                font.pixelSize: 9
                                                font.weight: Font.DemiBold
                                                font.letterSpacing: 0.5
                                            }
                                            MouseArea {
                                                id: installMouse
                                                anchors.fill: parent
                                                hoverEnabled: true
                                                cursorShape: Qt.PointingHandCursor
                                                onClicked: launcher.browse()
                                            }
                                        }
                                    }
                                }

                                Rectangle {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    radius: 11
                                    color: window.surface
                                    border.color: window.edge
                                    ColumnLayout {
                                        anchors.fill: parent
                                        anchors.margins: 15
                                        spacing: 10
                                        Text {
                                            text: "CONSOLE PROFILE"
                                            color: "#888b8f"
                                            font.pixelSize: 9
                                            font.weight: Font.Bold
                                            font.letterSpacing: 1.15
                                        }
                                        Text {
                                            text: launcher.consoleModel
                                            color: window.paper
                                            font.family: "Georgia"
                                            font.pixelSize: 18
                                        }
                                        RowLayout {
                                            Layout.fillWidth: true
                                            Layout.fillHeight: true
                                            spacing: 6
                                            Repeater {
                                                model: ["Series X", "Series S"]
                                                delegate: Rectangle {
                                                    required property string modelData
                                                    Layout.fillWidth: true
                                                    Layout.preferredHeight: 32
                                                    radius: 6
                                                    color: launcher.consoleModel === modelData ? "#3c2730" : "#1d2024"
                                                    border.color: launcher.consoleModel === modelData ? "#81505c" : "#363a3f"
                                                    Text {
                                                        anchors.centerIn: parent
                                                        text: modelData
                                                        color: launcher.consoleModel === modelData ? "#f0d6cf" : "#aaa9a5"
                                                        font.pixelSize: 10
                                                    }
                                                    MouseArea {
                                                        anchors.fill: parent
                                                        cursorShape: Qt.PointingHandCursor
                                                        onClicked: launcher.setConsoleModel(modelData)
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                Rectangle {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    radius: 11
                                    color: window.surface
                                    border.color: window.edge
                                    ColumnLayout {
                                        anchors.fill: parent
                                        anchors.margins: 15
                                        spacing: 9
                                        Text {
                                            text: "RECENT ACTIVITY"
                                            color: "#888b8f"
                                            font.pixelSize: 9
                                            font.weight: Font.Bold
                                            font.letterSpacing: 1.15
                                        }
                                        Text {
                                            Layout.fillWidth: true
                                            Layout.fillHeight: true
                                            text: launcher.logs.length > 0 ? launcher.logs[launcher.logs.length - 1] : "Your local launch history will appear here."
                                            color: window.paper
                                            font.family: "Georgia"
                                            font.pixelSize: 14
                                            wrapMode: Text.WordWrap
                                            elide: Text.ElideRight
                                        }
                                        Rectangle {
                                            Layout.preferredWidth: 114
                                            Layout.preferredHeight: 29
                                            radius: 6
                                            color: activityMouse.containsMouse ? "#282b30" : "#202328"
                                            border.color: "#373b40"
                                            Text {
                                                anchors.centerIn: parent
                                                text: "VIEW ACTIVITY"
                                                color: "#d9d3cb"
                                                font.pixelSize: 9
                                                font.weight: Font.DemiBold
                                                font.letterSpacing: 0.5
                                            }
                                            MouseArea {
                                                id: activityMouse
                                                anchors.fill: parent
                                                hoverEnabled: true
                                                cursorShape: Qt.PointingHandCursor
                                                onClicked: window.pageIndex = 1
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Item {
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 30
                        spacing: 16
                        RowLayout {
                            Layout.fillWidth: true
                            ColumnLayout {
                                spacing: 3
                                Text {
                                    text: "ACTIVITY"
                                    color: "#85888b"
                                    font.pixelSize: 9
                                    font.weight: Font.Bold
                                    font.letterSpacing: 1.6
                                }
                                Text {
                                    text: "Launch history"
                                    color: window.paper
                                    font.family: "Georgia"
                                    font.pixelSize: 27
                                }
                            }
                            Item { Layout.fillWidth: true }
                            Rectangle {
                                Layout.preferredWidth: 130
                                Layout.preferredHeight: 34
                                radius: 7
                                color: clearMouse.containsMouse ? "#25282d" : "#191b1f"
                                border.color: "#383c41"
                                Text {
                                    anchors.centerIn: parent
                                    text: "CLEAR HISTORY"
                                    color: "#d0cbc4"
                                    font.pixelSize: 9
                                    font.weight: Font.Bold
                                    font.letterSpacing: 0.8
                                }
                                MouseArea {
                                    id: clearMouse
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: launcher.clearActivity()
                                }
                            }
                        }
                        Text {
                            text: "GTA VI sessions and launcher messages stored on this PC."
                            color: window.muted
                            font.pixelSize: 12
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            radius: 12
                            color: window.surface
                            border.color: window.edge
                            ListView {
                                id: activityList
                                anchors.fill: parent
                                anchors.margins: 18
                                spacing: 10
                                clip: true
                                model: launcher.logs
                                onCountChanged: positionViewAtEnd()
                                delegate: Rectangle {
                                    required property string modelData
                                    required property int index
                                    width: activityList.width
                                    height: message.implicitHeight + 24
                                    radius: 8
                                    color: "#1c1f23"
                                    RowLayout {
                                        anchors.fill: parent
                                        anchors.margins: 12
                                        spacing: 10
                                        Rectangle {
                                            width: 6
                                            height: 6
                                            radius: 3
                                            color: index === activityList.count - 1 ? window.accent : "#74777b"
                                            Layout.alignment: Qt.AlignTop
                                            Layout.topMargin: 4
                                        }
                                        Text {
                                            id: message
                                            Layout.fillWidth: true
                                            text: modelData
                                            color: "#d3d0ca"
                                            font.family: "Cascadia Mono"
                                            font.pixelSize: 11
                                            wrapMode: Text.Wrap
                                        }
                                    }
                                }
                                Text {
                                    anchors.centerIn: parent
                                    visible: activityList.count === 0
                                    text: "No activity yet"
                                    color: "#92918c"
                                    font.family: "Georgia"
                                    font.pixelSize: 19
                                }
                            }
                        }
                    }
                }

                Item {
                    Flickable {
                        anchors.fill: parent
                        anchors.margins: 30
                        contentWidth: width
                        contentHeight: settingsContent.implicitHeight
                        clip: true
                        ColumnLayout {
                            id: settingsContent
                            width: parent.width
                            spacing: 14
                            Text {
                                text: "SETTINGS"
                                color: "#85888b"
                                font.pixelSize: 9
                                font.weight: Font.Bold
                                font.letterSpacing: 1.6
                            }
                            Text {
                                Layout.bottomMargin: 3
                                text: "Launcher preferences"
                                color: window.paper
                                font.family: "Georgia"
                                font.pixelSize: 27
                            }
                            Text {
                                Layout.bottomMargin: 5
                                text: "Manage your GTA VI profile, game files, and local data."
                                color: window.muted
                                font.pixelSize: 12
                            }
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 104
                                radius: 11
                                color: window.surface
                                border.color: window.edge
                                RowLayout {
                                    anchors.fill: parent
                                    anchors.margins: 18
                                    spacing: 13
                                    ColumnLayout {
                                        Layout.fillWidth: true
                                        spacing: 4
                                        Text {
                                            text: "CONSOLE PROFILE"
                                            color: "#b88f88"
                                            font.pixelSize: 9
                                            font.weight: Font.Bold
                                            font.letterSpacing: 1.2
                                        }
                                        Text {
                                            text: "New sessions use " + launcher.consoleModel + "."
                                            color: window.paper
                                            font.family: "Georgia"
                                            font.pixelSize: 17
                                        }
                                    }
                                    Repeater {
                                        model: ["Series X", "Series S"]
                                        delegate: Rectangle {
                                            required property string modelData
                                            Layout.preferredWidth: 102
                                            Layout.preferredHeight: 34
                                            radius: 7
                                            color: launcher.consoleModel === modelData ? "#3c2730" : "#1d2024"
                                            border.color: launcher.consoleModel === modelData ? "#81505c" : "#363a3f"
                                            Text {
                                                anchors.centerIn: parent
                                                text: modelData
                                                color: launcher.consoleModel === modelData ? "#f0d6cf" : "#aaa9a5"
                                                font.pixelSize: 10
                                            }
                                            MouseArea {
                                                anchors.fill: parent
                                                cursorShape: Qt.PointingHandCursor
                                                onClicked: launcher.setConsoleModel(modelData)
                                            }
                                        }
                                    }
                                }
                            }
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 142
                                radius: 11
                                color: window.surface
                                border.color: window.edge
                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 18
                                    spacing: 9
                                    Text {
                                        text: "GAME LOCATION"
                                        color: "#b88f88"
                                        font.pixelSize: 9
                                        font.weight: Font.Bold
                                        font.letterSpacing: 1.2
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        text: launcher.hasGame ? launcher.selectedPath : "Choose the GTA VI executable in your existing install folder."
                                        color: window.paper
                                        font.pixelSize: 11
                                        wrapMode: Text.WrapAnywhere
                                        maximumLineCount: 2
                                        elide: Text.ElideMiddle
                                    }
                                    Rectangle {
                                        Layout.preferredWidth: 164
                                        Layout.preferredHeight: 32
                                        radius: 7
                                        color: browseSettingsMouse.containsMouse ? "#282b30" : "#202328"
                                        border.color: "#373b40"
                                        Text {
                                            anchors.centerIn: parent
                                            text: launcher.hasGame ? "CHANGE LOCATION" : "LOCATE GTA VI"
                                            color: "#ddd6cf"
                                            font.pixelSize: 9
                                            font.weight: Font.DemiBold
                                            font.letterSpacing: 0.7
                                        }
                                        MouseArea {
                                            id: browseSettingsMouse
                                            anchors.fill: parent
                                            hoverEnabled: true
                                            cursorShape: Qt.PointingHandCursor
                                            onClicked: launcher.browse()
                                        }
                                    }
                                }
                            }
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 115
                                radius: 11
                                color: window.surface
                                border.color: window.edge
                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 18
                                    spacing: 8
                                    Text {
                                        text: "LOCAL DATA"
                                        color: "#b88f88"
                                        font.pixelSize: 9
                                        font.weight: Font.Bold
                                        font.letterSpacing: 1.2
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        text: launcher.dataDirectory
                                        color: window.paper
                                        font.family: "Cascadia Mono"
                                        font.pixelSize: 10
                                        wrapMode: Text.Wrap
                                    }
                                    Text {
                                        text: "Preferences and launch activity stay on this PC."
                                        color: window.muted
                                        font.pixelSize: 10
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}